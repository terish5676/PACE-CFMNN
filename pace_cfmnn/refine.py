"""Box edits and the acceptance gate for selective refinement (blueprint 7.2-7.3).

Operations on a labelled point x with true class t:

* A/B, add: ``CFMNN.add_points``, the model's own training rule (absorb into a
  same-class box under the size limit, otherwise a new point box).
* C, explanation-guided shrink: when x is misclassified and lies inside the
  winning wrong-class box k, move one face of k just past x on one feature,
  then add x. C picks the feature with the largest contrastive term e_i of
  "why the wrong class, not t" (``explain.contrastive_pair``).
* C', control shrink: the same edit with the feature chosen at random, or by
  the smallest share of the box's width cut ("volume").
* Exploratory choosers (refinement v2): "fewest" drops the fewest training
  members of the box; "pace_shap" takes the largest PACE-SHAP attribution to
  the wrong prediction. ``active.py`` also has an "oracle" cut that tries every
  feature through the gate (an upper bound, not a method).

After a shrink, box k keeps only the training members still inside it; its
Sum, Count and centroid are recomputed from them, and a box left empty is
removed. Only the main model's fixed centroid rule is supported: with the
paper's rule, Sum and Count do not match the members.

Every edit is checked by ``Gate`` on the acceptance rows and rolled back if
it fails. ``validate_structure`` checks the model's invariants; a failure
there is a bug, not a rejection, so it raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cfmnn_core import CFMNN, feature_terms
from .evaluate import classification_metrics
from .explain import contrastive_pair

MARGIN = 1e-3


def _require_fixed(model: CFMNN) -> None:
    if model.centroid_rule != "fixed":
        raise ValueError("refinement needs centroid_rule='fixed' (Sum/Count must match the members)")


def contains(model: CFMNN, k: int, x: np.ndarray) -> bool:
    """True if box k contains x (membership 1)."""
    return bool((model.V_[k] <= x).all() and (x <= model.W_[k]).all())


def shrink_target(model: CFMNN, x: np.ndarray, true_idx: int, det=None) -> int:
    """The winning box if it is of a wrong class and contains x, else -1."""
    det = det if det is not None else model.predict_details(x[None])
    k = int(det.winner[0])
    if int(det.cls[0]) == int(true_idx) or not contains(model, k, x):
        return -1
    return k


def face_cut(model: CFMNN, k: int, x: np.ndarray, i: int, margin: float = MARGIN):
    """Which face of box k to move on feature i, its new value, and the share of width cut.

    The face nearer to x moves just past x, so x falls outside the box. A share
    of 1 (or more, capped at 1) means the box would be cut away entirely.
    """
    v, w, xi = model.V_[k, i], model.W_[k, i], x[i]
    width = w - v
    if xi - v <= w - xi:
        new, side = xi + margin, "low"
        share = (new - v) / width if width > 0 else 1.0
    else:
        new, side = xi - margin, "high"
        share = (w - new) / width if width > 0 else 1.0
    return side, float(new), float(min(share, 1.0))


def shrink_box(model: CFMNN, k: int, x: np.ndarray, feature: int, X_lab: np.ndarray,
               margin: float = MARGIN) -> dict:
    """Move one face of box k past x on ``feature`` (in place). Returns what changed."""
    _require_fixed(model)
    side, new, share = face_cut(model, k, x, feature, margin)
    members = model.members_[k]
    if side == "low":
        gone = new > model.W_[k, feature]
        keep = members[X_lab[members, feature] >= new] if not gone else members[:0]
    else:
        gone = new < model.V_[k, feature]
        keep = members[X_lab[members, feature] <= new] if not gone else members[:0]
    info = {"box": int(k), "feature": int(feature), "side": side, "share_cut": share,
            "n_cut": int(len(members) - len(keep)), "removed": bool(len(keep) == 0)}
    if len(keep) == 0:
        model._remove_box(k)
        return info
    if side == "low":
        model.V_[k, feature] = new
    else:
        model.W_[k, feature] = new
    model.members_[k] = keep
    model.sum_[k] = X_lab[keep].sum(axis=0)
    model.count_[k] = len(keep)
    model.centroid_[k] = model.sum_[k] / len(keep)
    return info


# ------------------------------------------------------------ feature choice
# Every chooser takes (model, x, k, wrong_idx, true_idx, rng) and may use the
# keyword arguments X_lab (the labelled rows that members_ index) and
# background (reference rows for PACE-SHAP); the others ignore them.
def feature_explained(model, x, k, wrong_idx, true_idx, rng=None, **_) -> int:
    """C: the largest contrastive term of 'why the wrong class, not the true one'."""
    c = contrastive_pair(model, x[None], np.array([wrong_idx]), np.array([true_idx]))
    return int(np.argmax(c.e[0]))


def feature_random(model, x, k, wrong_idx, true_idx, rng=None, **_) -> int:
    """C' control: a random feature."""
    return int(rng.integers(model.n_features_in_))


def feature_volume(model, x, k, wrong_idx, true_idx, rng=None, **_) -> int:
    """C' control: the feature where the cut removes the smallest share of the box's width."""
    shares = [face_cut(model, k, x, i)[2] for i in range(model.n_features_in_)]
    return int(np.argmin(shares))


def members_cut(model: CFMNN, k: int, x: np.ndarray, i: int, X_lab: np.ndarray) -> int:
    """How many training members of box k the cut on feature i would drop."""
    side, new, _ = face_cut(model, k, x, i)
    vals = X_lab[model.members_[k], i]
    return int(np.sum(vals < new) if side == "low" else np.sum(vals > new))


def feature_fewest_members(model, x, k, wrong_idx, true_idx, rng=None, X_lab=None, **_) -> int:
    """Data-aware control: the feature whose cut drops the fewest training members of box k.

    Ties go to the smaller share of width cut, then to the lower feature index.
    """
    if X_lab is None:
        raise ValueError("feature_fewest_members needs X_lab")
    d = model.n_features_in_
    key = [(members_cut(model, k, x, i, X_lab), face_cut(model, k, x, i)[2], i) for i in range(d)]
    return int(min(key)[2])


def feature_pace_shap(model, x, k, wrong_idx, true_idx, rng=None, background=None, **_) -> int:
    """C with PACE-SHAP: the feature that pushes the (wrong) prediction's margin up the most.

    PACE-SHAP explains the predicted class's margin over the runner-up, which
    is the wrong class here, so the largest attribution is the feature most
    responsible for the error.
    """
    from .explain import PaceShap   # local import: explain imports nothing from here
    if background is None:
        raise ValueError("feature_pace_shap needs background rows")
    phi = PaceShap(model, background, seed=0).explain(x[None])[0]
    return int(np.argmax(phi))


CHOOSERS = {"explained": feature_explained, "random": feature_random, "volume": feature_volume,
            "fewest": feature_fewest_members, "pace_shap": feature_pace_shap}


# ------------------------------------------------------------------- the gate
@dataclass
class Gate:
    """Accept an edit only if the acceptance rows show no meaningful harm (blueprint 7.3).

    Thresholds are in percentage points: macro-F1 may fall by at most
    ``max_f1_drop`` and no class's recall by more than ``max_recall_drop``,
    each compared with the model before the edit. The box count must stay at
    or below ``box_cap``.
    """

    X: np.ndarray
    y: np.ndarray
    box_cap: int
    max_f1_drop: float = 0.5
    max_recall_drop: float = 1.0

    def measure(self, model: CFMNN) -> dict:
        return classification_metrics(self.y, model.predict(self.X))

    def verdict(self, before: dict, after: dict) -> tuple[bool, str]:
        """(accepted, reason) for the gate-row metrics after an edit, box cap aside."""
        if before["macro_f1"] - after["macro_f1"] > self.max_f1_drop + 1e-9:
            return False, "macro-F1"
        for key, val in before.items():
            if key.startswith("recall_") and val - after[key] > self.max_recall_drop + 1e-9:
                return False, "class recall"
        return True, ""

    def check(self, before: dict, model: CFMNN):
        """(accepted, reason, metrics after the edit)."""
        if model.n_boxes_ > self.box_cap:
            return False, "box cap", None
        after = self.measure(model)
        ok, reason = self.verdict(before, after)
        return ok, reason, after

    def peek(self, model: CFMNN):
        """(class index predicted for each gate row, gate metrics) for ``model``, changing nothing.

        Used to score a trial edit without committing it (``active.run_self_repair``'s
        greedy order).
        """
        pred = np.asarray(model.predict(self.X))
        return np.searchsorted(model.classes_, pred), classification_metrics(self.y, pred)


@dataclass
class CachedGate(Gate):
    """The same gate, with the same decisions, but cheaper after a one-box edit.

    ``Gate.measure`` predicts every gate row after every edit (n rows x H boxes x d
    features). A self-repair edit changes one box: it moves one face, or removes the box.
    This gate keeps, for the gate rows, the membership table M (n x H), the squared
    centroid distances D (n x H), each row's top membership and predicted class, and
    copies of V, W, the centroids and the box classes. Both tables are built with the
    expressions of ``CFMNN.predict_details``.

    On each ``measure`` it compares the model with its copies:

    * nothing changed: the cached predictions stand;
    * one box moved (V, W or centroid changed; same box class): recompute that box's
      column of M and D, and re-score only the rows where the box was in the tied top
      group before (M >= top - tol), or reaches it now. Every other row keeps its top
      membership, its tied set and its distances, so its prediction cannot change;
    * one box removed: delete its column, and re-score the rows where it was tied at the top;
    * anything else (several boxes, a changed box class, added boxes): full recompute.

    ``check`` undoes the cache update when the edit is rejected, which matches the caller
    restoring the model from its snapshot.
    """

    _c: dict | None = field(default=None, init=False, repr=False)
    _undo: tuple | None = field(default=None, init=False, repr=False)

    @staticmethod
    def _score(M, D, top, box_class, tol) -> np.ndarray:
        """Predicted box class of each row: top membership, ties to the nearest centroid."""
        tied = M >= top[:, None] - tol
        return box_class[np.argmin(np.where(tied, D, np.inf), axis=1)]

    def _full(self, model: CFMNN) -> dict:
        X, tol = self.X, float(model.tie_tol)
        n, H = len(X), model.n_boxes_
        M, D = np.empty((n, H)), np.empty((n, H))
        step = model._chunk_size()
        for s in range(0, n, step):
            Xc = X[s:s + step]
            M[s:s + step] = feature_terms(Xc[:, None, :], model.V_, model.W_, model.gamma).mean(axis=2)
            D[s:s + step] = ((Xc[:, None, :] - model.centroid_[None, :, :]) ** 2).sum(axis=2)
        top = M.max(axis=1)
        return {"V": model.V_.copy(), "W": model.W_.copy(), "cen": model.centroid_.copy(),
                "bc": model.box_class_.copy(), "gamma": model.gamma, "tol": tol,
                "classes": np.asarray(model.classes_).copy(), "M": M, "D": D, "top": top,
                "cls": self._score(M, D, top, model.box_class_, tol)}

    def _removed_index(self, model: CFMNN):
        c = self._c
        diff = ((c["V"][:-1] != model.V_).any(axis=1) | (c["W"][:-1] != model.W_).any(axis=1)
                | (c["cen"][:-1] != model.centroid_).any(axis=1) | (c["bc"][:-1] != model.box_class_))
        k = int(np.argmax(diff)) if diff.any() else len(c["V"]) - 1
        for name, now in (("V", model.V_), ("W", model.W_), ("cen", model.centroid_),
                          ("bc", model.box_class_)):
            if not np.array_equal(np.delete(c[name], k, axis=0), now):
                return None
        return k

    def _move(self, model: CFMNN, k: int) -> None:
        c, X, tol = self._c, self.X, self._c["tol"]
        old_top = c["top"]
        was_tied = c["M"][:, k] >= old_top - tol
        self._undo = ("move", k, c["M"][:, k].copy(), c["D"][:, k].copy(), c["V"][k].copy(),
                      c["W"][k].copy(), c["cen"][k].copy(), c["top"], c["cls"])
        Mk = feature_terms(X[:, None, :], model.V_[k:k + 1], model.W_[k:k + 1], model.gamma).mean(axis=2)[:, 0]
        Dk = ((X - model.centroid_[k]) ** 2).sum(axis=1)
        rows = np.flatnonzero(was_tied | (Mk >= old_top - tol))
        c["M"][:, k], c["D"][:, k] = Mk, Dk
        top, cls = old_top.copy(), c["cls"].copy()
        if len(rows):
            top_r = c["M"][rows].max(axis=1)
            top[rows] = top_r
            cls[rows] = self._score(c["M"][rows], c["D"][rows], top_r, model.box_class_, tol)
        c["top"], c["cls"] = top, cls
        c["V"][k], c["W"][k], c["cen"][k] = model.V_[k], model.W_[k], model.centroid_[k]

    def _remove(self, model: CFMNN, k: int) -> None:
        c, tol = self._c, self._c["tol"]
        self._undo = ("all", c)
        M, D = np.delete(c["M"], k, axis=1), np.delete(c["D"], k, axis=1)
        top, cls = c["top"].copy(), c["cls"].copy()
        rows = np.flatnonzero(c["M"][:, k] >= c["top"] - tol)
        if len(rows):
            top_r = M[rows].max(axis=1)
            top[rows] = top_r
            cls[rows] = self._score(M[rows], D[rows], top_r, model.box_class_, tol)
        self._c = {**c, "V": model.V_.copy(), "W": model.W_.copy(), "cen": model.centroid_.copy(),
                   "bc": model.box_class_.copy(), "M": M, "D": D, "top": top, "cls": cls}

    def _sync(self, model: CFMNN) -> None:
        c = self._c
        self._undo = None
        if (c is None or c["gamma"] != model.gamma or c["tol"] != float(model.tie_tol)
                or not np.array_equal(c["classes"], model.classes_)):
            self._c = self._full(model)
            return
        H0, H1 = len(c["V"]), model.n_boxes_
        if H1 == H0:
            diff = ((c["V"] != model.V_).any(axis=1) | (c["W"] != model.W_).any(axis=1)
                    | (c["cen"] != model.centroid_).any(axis=1) | (c["bc"] != model.box_class_))
            moved = np.flatnonzero(diff)
            if len(moved) == 0:
                return
            if len(moved) == 1 and c["bc"][moved[0]] == model.box_class_[moved[0]]:
                self._move(model, int(moved[0]))
                return
        elif H1 == H0 - 1:
            k = self._removed_index(model)
            if k is not None:
                self._remove(model, k)
                return
        self._undo = ("all", c)
        self._c = self._full(model)

    def measure(self, model: CFMNN) -> dict:
        self._sync(model)
        return classification_metrics(self.y, self._c["classes"][self._c["cls"]])

    def _restore(self) -> None:
        """Undo the last cache update (``_undo`` must be set)."""
        u = self._undo
        if u[0] == "all":
            self._c = u[1]
        else:
            _, k, Mk, Dk, v, w, cen, top, cls = u
            c = self._c
            c["M"][:, k], c["D"][:, k] = Mk, Dk
            c["V"][k], c["W"][k], c["cen"][k] = v, w, cen
            c["top"], c["cls"] = top, cls

    def check(self, before: dict, model: CFMNN):
        self._undo = None
        ok, reason, after = super().check(before, model)
        if not ok and self._undo is not None:
            self._restore()
        self._undo = None
        return ok, reason, after

    def peek(self, model: CFMNN):
        """As ``Gate.peek``, from the cache; the cache is left as it was."""
        old = self._c
        self._undo = None
        self._sync(model)
        cls = self._c["cls"].copy()
        metrics = classification_metrics(self.y, self._c["classes"][cls])
        if self._undo is not None:
            self._restore()
        else:               # nothing changed (same tables), or a first build: drop it
            self._c = old
        self._undo = None
        return cls, metrics


def validate_structure(model: CFMNN, X_lab: np.ndarray, y_lab: np.ndarray) -> None:
    """Raise if any invariant is broken: bounds, Sum/Count/centroid vs members, member classes."""
    _require_fixed(model)
    if not ((model.V_ <= model.W_).all() and (model.V_ >= 0).all() and (model.W_ <= 1).all()):
        raise ValueError("box bounds are invalid")
    if len(model.members_) != model.n_boxes_:
        raise ValueError("members_ does not match the boxes")
    seen = np.concatenate(model.members_) if model.n_boxes_ else np.array([], int)
    if len(np.unique(seen)) != len(seen):
        raise ValueError("a training row belongs to two boxes")
    for j, m in enumerate(model.members_):
        P = X_lab[m]
        if model.count_[j] != len(m) or len(m) == 0:
            raise ValueError(f"box {j}: count does not match its members")
        if not np.allclose(model.sum_[j], P.sum(axis=0), atol=1e-8):
            raise ValueError(f"box {j}: sum does not match its members")
        if not np.allclose(model.centroid_[j], model.sum_[j] / model.count_[j], atol=1e-12):
            raise ValueError(f"box {j}: centroid is stale")
        if not ((P >= model.V_[j]).all() and (P <= model.W_[j]).all()):
            raise ValueError(f"box {j}: a member lies outside the box")
        if not (y_lab[m] == model.classes_[model.box_class_[j]]).all():
            raise ValueError(f"box {j}: a member has another class")
    for c, idx in enumerate(model.class_boxes_):
        if not np.array_equal(idx, np.flatnonzero(model.box_class_ == c)):
            raise ValueError("class_boxes_ is stale")
