"""Budgeted active-learning loop for selective refinement (blueprint 7.1).

The pool's labels sit behind a ``LabelOracle`` and are revealed only for the
points a selector picks. Each round:

1. score the remaining pool with the current model and pick the top b points
   (ties broken at random, seeded);
2. reveal their labels;
3. for each point, try its candidate edits in order: the shrink (if the arm
   has one and the point is misclassified inside a wrong-class box), then the
   plain add. Each is gated on the acceptance rows; a failed edit is rolled
   back and the next is tried. A point whose edits all fail is labelled but
   left out of the model.

Test rows, if given, are only measured after each round for the learning
curve. Nothing in the loop reads them.

Selectors (higher = picked first):

* ``random``;
* ``margin``: the plain margin uncertainty 1 - (s'_(1) - s'_(2));
* ``pd_u_star``: PD + U*, percentile ranks fitted on the inner rows with the
  current model (EI is skipped; it adds nothing, see RQ2);
* ``order_ensemble``: the share of 5 CFMNNs, trained on the current labelled
  data in other orders, that disagree with the current model;
* ``knn`` (refinement v2): the share of the 10 nearest labelled rows whose
  label differs from the prediction (RQ2's strongest single detector);
* ``oracle_errors`` (refinement v2, reference only): reads the pool's hidden
  labels and picks the points the model gets wrong. It is an upper bound on
  what any selector could do, not a method.

``run_self_repair`` (refinement v2) runs the shrink on the training rows the
model gets wrong. Their labels are already known, so it buys no new labels.

Every round log counts labels: ``pool_labels`` bought from the oracle,
``gate_labels`` read by the gate (the acceptance rows, reused for every edit),
and ``total_labels``.
"""

from __future__ import annotations

import copy
import heapq
import time
from dataclasses import dataclass

import numpy as np

from .cfmnn_core import CFMNN
from .evaluate import classification_metrics
from .neighborhood import Perturber
from .pace_score import PRS
from .refine import CHOOSERS, CachedGate, Gate, shrink_box, shrink_target, validate_structure
from .signals import SignalComputer, uncertainty
from .trust_baselines import knn_disagreement, order_ensemble_disagreement

# Arm name -> feature chooser for the shrink (None = add only). "oracle" tries
# every feature through the gate and keeps the best (an upper bound).
OPERATIONS = {"A+B": None, "A+B+C": "explained", "A+B+C'-random": "random",
              "A+B+C'-volume": "volume", "A+B+C-fewest": "fewest",
              "A+B+C-paceshap": "pace_shap", "A+B+C-oracle": "oracle"}


class LabelOracle:
    """Holds the pool's labels; reveals each one once, and logs every read."""

    def __init__(self, y):
        self._y = np.asarray(y)
        self._log: list[int] = []

    def reveal(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if len(np.unique(idx)) != len(idx) or np.isin(idx, self._log).any():
            raise ValueError("each pool label may be revealed only once")
        self._log.extend(idx.tolist())
        return self._y[idx].copy()

    @property
    def revealed(self) -> np.ndarray:
        return np.asarray(self._log, dtype=np.int64)

    def peek_for_reference(self, idx) -> np.ndarray:
        """Hidden labels for the ``oracle_errors`` reference selector only. Not a purchase."""
        return self._y[np.asarray(idx, dtype=np.int64)].copy()


@dataclass
class Context:
    """What a selector may see. No pool labels, no acceptance or test rows.

    ``y_cand_hidden`` is filled only for the ``oracle_errors`` reference arm.
    """

    model: CFMNN
    X_cand: np.ndarray
    X_lab: np.ndarray
    y_lab: np.ndarray
    X_inner: np.ndarray
    perturber: Perturber | None
    rng: np.random.Generator
    seed: int
    y_cand_hidden: np.ndarray | None = None


def select_random(ctx: Context) -> np.ndarray:
    return ctx.rng.uniform(size=len(ctx.X_cand))


def select_margin(ctx: Context) -> np.ndarray:
    return uncertainty(ctx.model, ctx.model.predict_details(ctx.X_cand))[2]


def select_pd_u_star(ctx: Context) -> np.ndarray:
    sc = SignalComputer(ctx.model, ctx.perturber, ei=False)
    prs = PRS().fit(sc.compute(ctx.X_inner, seed=10_000 + ctx.seed))
    return prs.score(sc.compute(ctx.X_cand, seed=ctx.seed), use=("pd", "u_star"))[0]


def select_order_ensemble(ctx: Context) -> np.ndarray:
    pred = ctx.model.predict(ctx.X_cand)
    return order_ensemble_disagreement(ctx.model, ctx.X_lab, ctx.y_lab, ctx.X_cand, pred,
                                       n_models=5, seed=ctx.seed)


def select_knn(ctx: Context) -> np.ndarray:
    pred = ctx.model.predict(ctx.X_cand)
    return knn_disagreement(ctx.X_lab, ctx.y_lab, ctx.X_cand, pred, k=10)


def select_oracle_errors(ctx: Context) -> np.ndarray:
    if ctx.y_cand_hidden is None:
        raise ValueError("oracle_errors needs the hidden pool labels")
    return (ctx.model.predict(ctx.X_cand) != ctx.y_cand_hidden).astype(float)


SELECTORS = {"random": select_random, "margin": select_margin, "pd_u_star": select_pd_u_star,
             "order_ensemble": select_order_ensemble, "knn": select_knn,
             "oracle_errors": select_oracle_errors}


def top_b(scores: np.ndarray, b: int, rng: np.random.Generator) -> np.ndarray:
    """Positions of the b highest scores; ties broken by a seeded random order."""
    order = rng.permutation(len(scores))
    return order[np.argsort(-np.asarray(scores)[order], kind="stable")][:b]


def _oracle_cut(snap: CFMNN, k: int, x, t, row: int, X_lab, gate: Gate, before: dict):
    """Try the shrink on every feature (each followed by the add) and keep the gate's best.

    The best accepted trial has the highest acceptance macro-F1, then the
    smallest share of width cut, then the lowest feature index. Returns
    (model or None, info, metrics after or None).
    """
    best = None
    for i in range(snap.n_features_in_):
        trial = copy.deepcopy(snap)
        info = shrink_box(trial, k, x, i, X_lab)
        trial.add_points(x, [t], rows=[row])
        ok, _, after = gate.check(before, trial)
        if ok:
            key = (after["macro_f1"], -info["share_cut"], -i)
            if best is None or key > best[0]:
                best = (key, trial, info, after)
    if best is None:
        return None, {}, None
    return best[1], best[2], best[3]


def run_refinement(model: CFMNN, X_train, y_train, X_pool, oracle: LabelOracle, X_inner,
                   X_acc, y_acc, selector: str, operation: str = "A+B", budget: int = 300,
                   n_rounds: int = 10, seed: int = 0, perturber: Perturber | None = None,
                   X_test=None, y_test=None, box_cap_factor: float = 1.1,
                   max_f1_drop: float = 0.5, max_recall_drop: float = 1.0,
                   background=None, on_accept=None):
    """Run one arm. ``model`` must be fitted on ``X_train`` in this row order; it is not changed.

    ``background`` holds reference rows for the ``pace_shap`` chooser.
    ``on_accept(before, after, removed)`` is called after every accepted edit,
    with the model before and after it and the index of a box the shrink
    deleted (or None); ``audit.impact_set`` uses it.

    Returns (refined model, per-round log, per-edit log).
    """
    if selector not in SELECTORS or operation not in OPERATIONS:
        raise ValueError("unknown selector or operation")
    if selector == "pd_u_star" and perturber is None:
        raise ValueError("pd_u_star needs a fitted Perturber")
    model = copy.deepcopy(model)
    chooser = OPERATIONS[operation]
    rng = np.random.default_rng(seed)
    X_lab = np.array(X_train, dtype=float)
    y_lab = np.asarray(y_train).copy()
    X_pool = np.asarray(X_pool, dtype=float)
    validate_structure(model, X_lab, y_lab)
    gate = Gate(np.asarray(X_acc, float), np.asarray(y_acc), int(np.floor(box_cap_factor * model.n_boxes_)),
                max_f1_drop, max_recall_drop)
    before = gate.measure(model)
    b = int(np.ceil(budget / n_rounds))
    remaining = np.arange(len(X_pool))
    rounds, edits = [], []
    n_gate = len(np.asarray(y_acc))

    def log_round(r: int, used: int) -> None:
        rec = {"round": r, "labels": used, "pool_labels": used, "gate_labels": n_gate,
               "total_labels": used + n_gate, "boxes": model.n_boxes_,
               "accept_macro_f1": before["macro_f1"]}
        if X_test is not None:
            rec.update({f"test_{k}": v for k, v in
                        classification_metrics(y_test, model.predict(X_test)).items()})
        rounds.append(rec)

    log_round(0, 0)
    used = 0
    for r in range(1, n_rounds + 1):
        take = min(b, budget - used, len(remaining))
        if take <= 0:
            break
        ctx = Context(model, X_pool[remaining], X_lab, y_lab, np.asarray(X_inner, float),
                      perturber, rng, seed * 1000 + r)
        if selector == "oracle_errors":
            ctx.y_cand_hidden = oracle.peek_for_reference(remaining)
        pick = remaining[top_b(SELECTORS[selector](ctx), take, rng)]
        labels = oracle.reveal(pick)
        remaining = remaining[~np.isin(remaining, pick)]
        for x, t, p in zip(X_pool[pick], labels, pick):
            row = len(X_lab)
            X_lab = np.vstack([X_lab, x])
            y_lab = np.append(y_lab, t)
            det = model.predict_details(x[None])
            t_idx = int(np.searchsorted(model.classes_, t))
            wrong = int(det.cls[0]) != t_idx
            k = shrink_target(model, x, t_idx, det) if wrong else -1
            cands = (["shrink"] if (chooser and k >= 0) else []) + ["add"]
            for cand in cands:
                snap = copy.deepcopy(model)
                info = {}
                if cand == "shrink" and chooser == "oracle":
                    trial, info, after = _oracle_cut(snap, k, x, t, row, X_lab, gate, before)
                    ok, reason = trial is not None, "" if trial is not None else "no feature passes"
                    if ok:
                        model = trial
                else:
                    if cand == "shrink":
                        i = CHOOSERS[chooser](model, x, k, int(det.cls[0]), t_idx, rng,
                                              X_lab=X_lab, background=background)
                        info = shrink_box(model, k, x, i, X_lab)
                    model.add_points(x, [t], rows=[row])
                    ok, reason, after = gate.check(before, model)
                edits.append({"round": r, "pool_index": int(p), "edit": cand, "was_wrong": wrong,
                              "shrink_possible": k >= 0, "accepted": ok, "reason": reason,
                              "x_correct_after": bool(model.predict(x[None])[0] == t),
                              **{f"shrink_{a}": v for a, v in info.items()}})
                if ok:
                    before = after
                    if on_accept is not None:
                        on_accept(snap, model, info.get("box") if info.get("removed") else None)
                    break
                model = snap
            used += 1
        validate_structure(model, X_lab, y_lab)
        log_round(r, used)
    return model, rounds, edits


ORDERS = ("random", "box", "greedy")


def _trial_copy(model: CFMNN) -> CFMNN:
    """A copy that ``shrink_box`` can change without touching ``model``.

    The per-box arrays and the member list are copied; the member arrays themselves are
    shared, because ``shrink_box`` and ``_remove_box`` replace them and never edit them.
    Much cheaper than ``copy.deepcopy`` for scoring trial edits.
    """
    m = copy.copy(model)
    for name in ("V_", "W_", "sum_", "count_", "centroid_", "box_class_"):
        setattr(m, name, getattr(model, name).copy())
    m.members_ = list(model.members_)
    m.class_boxes_ = list(model.class_boxes_)
    return m


def box_order(model: CFMNN, X, perm: np.ndarray) -> np.ndarray:
    """``perm`` (training-error rows) grouped by the box that wins them, biggest group first.

    Groups of equal size keep the order in which they first appear in ``perm``, and so
    do the rows inside a group.
    """
    win = np.asarray(model.predict_details(np.asarray(X)[perm]).winner)
    _, first, inv, size = np.unique(win, return_index=True, return_inverse=True, return_counts=True)
    return perm[np.lexsort((np.arange(len(perm)), first[inv], -size[inv]))]


def run_self_repair(model: CFMNN, X_train, y_train, X_gate, y_gate, chooser: str = "volume",
                    budget: int = 300, n_rounds: int = 10, seed: int = 0, X_test=None, y_test=None,
                    box_cap_factor: float = 1.1, max_f1_drop: float = 0.5,
                    max_recall_drop: float = 1.0, gate_labels_are_new: bool = True,
                    background=None, on_accept=None, fast_gate: bool = False,
                    gate_reference: str = "last", order: str = "random",
                    count_skips: bool = True, on_try=None):
    """Repair the model on its own training errors: no new labels are bought.

    ``fast_gate=True`` uses ``refine.CachedGate``, which makes the same decisions as the
    standard gate. ``gate_reference="start"`` compares every edit with the model before any
    edit (a cumulative gate) instead of with the model before that edit ("last", the default).

    Every training row the model gets wrong lies inside a box of its own class
    and inside a box of the wrong class, and the centroid rule picks the wrong
    one (checked on the v1 datasets). So the shrink can be applied to these
    rows directly: move one face of the winning wrong-class box past the row,
    then gate the edit. The row is already a member of its own box, so nothing
    is added.

    Up to ``budget`` training errors are tried in ``n_rounds`` rounds; a row the
    model already gets right by its turn is skipped (the try still counts). The
    gate reads ``X_gate``, ``y_gate``: the acceptance rows, or the training rows
    (pass ``gate_labels_are_new=False``, so no labels are counted).

    ``order`` sets which errors are tried, and when (Run H):

    * "random" (the default, and every committed run): a seeded random order;
    * "box": the same errors grouped by the wrong-class box that wins them at the
      start, the biggest group first (``box_order``);
    * "greedy": at the start of each round, every error not yet tried is scored by
      a trial edit on the current model, the same cut the real try would make: the
      gate rows it would fix minus those it would break. Edits the gate would pass
      rank first, then edits it would reject, then rows that cannot be edited (already
      right, or not inside the winning box); the highest score goes first. Before each
      try, an error scored on an older model is re-scored and taken only if it still
      ranks first. Ties follow the random order. Needs a chooser that uses no random
      numbers.

    ``count_skips`` (Run H, amendment 1): with True (the default, and every committed
    run), a try on a row that is already right, or not inside the winning box, counts
    against ``budget``. With False ("real tries"), such rows are skipped without counting,
    and the random or box order runs on until ``budget`` edits have reached the gate (or
    the errors run out); each round gates as many edits as a round of the default order
    tries. Not for the greedy order, which already puts such rows last.

    ``on_try(before, after, info, accepted, gate_seconds)``, if given, is called after every
    gated edit, accepted or not, before a rejected edit is rolled back (Run P11). It must not
    change the models.

    Returns (repaired model, per-round log, per-edit log).
    """
    if chooser not in CHOOSERS:
        raise ValueError("unknown chooser")
    if gate_reference not in ("last", "start"):
        raise ValueError("gate_reference must be 'last' or 'start'")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    if order == "greedy" and chooser == "random":
        raise ValueError("the greedy order needs a chooser that uses no random numbers")
    if order == "greedy" and not count_skips:
        raise ValueError("count_skips=False is for the random and box orders")
    model = copy.deepcopy(model)
    rng = np.random.default_rng(seed)
    X_lab = np.array(X_train, dtype=float)
    y_lab = np.asarray(y_train).copy()
    validate_structure(model, X_lab, y_lab)
    gate = (CachedGate if fast_gate else Gate)(
        np.asarray(X_gate, float), np.asarray(y_gate),
        int(np.floor(box_cap_factor * model.n_boxes_)), max_f1_drop, max_recall_drop)
    before = gate.measure(model)
    t_all = np.searchsorted(model.classes_, y_lab)
    errors = np.flatnonzero(model.predict_details(X_lab).cls != t_all)
    perm = rng.permutation(errors)
    n_gate = len(np.asarray(y_gate)) if gate_labels_are_new else 0
    rounds, edits = [], []
    n_trials = 0

    def log_round(r: int, tried: int) -> None:
        rec = {"round": r, "labels": 0, "pool_labels": 0, "gate_labels": n_gate,
               "total_labels": n_gate, "train_errors_tried": tried, "boxes": model.n_boxes_,
               "accept_macro_f1": before["macro_f1"]}
        if order == "greedy":
            rec["trial_edits"] = n_trials
        if X_test is not None:
            rec.update({f"test_{k}": v for k, v in
                        classification_metrics(y_test, model.predict(X_test)).items()})
        rounds.append(rec)

    def attempt(p, r: int, extra: dict) -> bool:
        """Try training error p once (or skip it). True if an edit was accepted."""
        nonlocal model, before
        x, t_idx = X_lab[p], int(t_all[p])
        det = model.predict_details(x[None])
        rec = {"round": r, "train_index": int(p), "edit": "shrink", **extra}
        if int(det.cls[0]) == t_idx:
            edits.append({**rec, "edit": "none", "accepted": False, "reason": "already right"})
            return False
        k = shrink_target(model, x, t_idx, det)
        if k < 0:
            edits.append({**rec, "edit": "none", "accepted": False, "reason": "not inside the winning box"})
            return False
        snap = copy.deepcopy(model)
        i = CHOOSERS[chooser](model, x, k, int(det.cls[0]), t_idx, rng,
                              X_lab=X_lab, background=background)
        info = shrink_box(model, k, x, i, X_lab)
        t_gate = time.perf_counter()
        ok, reason, after = gate.check(before, model)
        if on_try is not None:
            on_try(snap, model, info, ok, time.perf_counter() - t_gate)
        edits.append({**rec, "accepted": ok, "reason": reason,
                      "x_correct_after": bool(model.predict(x[None])[0] == y_lab[p]),
                      **{f"shrink_{a}": v for a, v in info.items()}})
        if ok:
            if gate_reference == "last":
                before = after
            if on_accept is not None:
                on_accept(snap, model, info["box"] if info["removed"] else None)
        else:
            model = snap
        return ok

    log_round(0, 0)
    tried = 0
    if order != "greedy" and count_skips:
        seq = (perm if order == "random" else box_order(model, X_lab, perm))[:budget]
        for r, chunk in enumerate(np.array_split(seq, n_rounds), start=1):
            for p in chunk:
                tried += 1
                attempt(p, r, {})
            validate_structure(model, X_lab, y_lab)
            log_round(r, tried)
        return model, rounds, edits

    if order != "greedy":
        # real tries: skips do not count; each round gates a default round's number of edits
        seq = iter(perm if order == "random" else box_order(model, X_lab, perm))
        quotas = [len(c) for c in np.array_split(np.arange(min(budget, len(perm))), n_rounds)]
        for r, quota in enumerate(quotas, start=1):
            gated = 0
            while gated < quota:
                p = next(seq, None)
                if p is None:
                    break
                tried += 1
                attempt(p, r, {})
                gated += edits[-1]["edit"] == "shrink"
            validate_structure(model, X_lab, y_lab)
            log_round(r, tried)
        return model, rounds, edits

    # greedy: lazy re-scoring; a score is fresh while no edit has been accepted since it
    y_gate_idx = np.searchsorted(model.classes_, np.asarray(y_gate))
    where = {int(p): j for j, p in enumerate(perm)}
    left = [int(p) for p in perm]
    version = 0
    now = None                                  # (version, gate rows right under the model)

    def right_now() -> int:
        nonlocal now
        if now is None or now[0] != version:
            cls, _ = gate.peek(model)
            now = (version, int(np.sum(cls == y_gate_idx)))
        return now[1]

    def key(p: int) -> tuple:
        """Sort key of error p on the current model (smaller is better)."""
        nonlocal n_trials
        x, t_idx = X_lab[p], int(t_all[p])
        det = model.predict_details(x[None])
        if int(det.cls[0]) == t_idx:
            return (2, 0)
        k = shrink_target(model, x, t_idx, det)
        if k < 0:
            return (2, 0)
        trial = _trial_copy(model)
        i = CHOOSERS[chooser](trial, x, k, int(det.cls[0]), t_idx, None,
                              X_lab=X_lab, background=background)
        shrink_box(trial, k, x, i, X_lab)
        n_trials += 1
        cls, after = gate.peek(trial)
        gain = int(np.sum(cls == y_gate_idx)) - right_now()
        ok = trial.n_boxes_ <= gate.box_cap and gate.verdict(before, after)[0]
        return (0 if ok else 1, -gain)

    quotas = [len(c) for c in np.array_split(np.arange(min(budget, len(perm))), n_rounds)]
    for r, quota in enumerate(quotas, start=1):
        heap = [(key(p), where[p], version, p) for p in left]
        heapq.heapify(heap)
        taken = 0
        while taken < quota and heap:
            sk, w, v, p = heapq.heappop(heap)
            if v != version:
                sk = key(p)
                if heap and (sk, w) > heap[0][:2]:
                    heapq.heappush(heap, (sk, w, version, p))
                    continue
            tried += 1
            taken += 1
            left.remove(p)
            if attempt(p, r, {"greedy_class": sk[0], "greedy_gain": -sk[1] if sk[0] < 2 else np.nan}):
                version += 1
        validate_structure(model, X_lab, y_lab)
        log_round(r, tried)
    return model, rounds, edits
