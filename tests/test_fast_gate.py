"""Run E: the cached gate must make exactly the standard gate's decisions."""

import copy
import inspect
import unittest

import numpy as np

from pace_cfmnn import CFMNN, data
from pace_cfmnn.active import run_self_repair
from pace_cfmnn.evaluate import classification_metrics
from pace_cfmnn.refine import CachedGate, Gate, shrink_box, validate_structure

ARRAYS = ("V_", "W_", "sum_", "count_", "box_class_", "centroid_")


def problem(seed=0, n=700, d=5, classes=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    y = ((X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.15, n)) > 0.75).astype(int)
    if classes == 3:
        y = y + (X[:, 2] > 0.8)
    return X, y


def same_model(a, b):
    return (all(np.array_equal(getattr(m, k), getattr(b, k)) for m in (a,) for k in ARRAYS)
            and len(a.members_) == len(b.members_)
            and all(np.array_equal(p, q) for p, q in zip(a.members_, b.members_)))


def assert_cache_is_exact(case, g, model):
    """The cached tables equal a fresh computation, and the predictions equal model.predict."""
    fresh = g._full(model)
    for key in ("V", "W", "cen", "bc", "M", "D", "top", "cls"):
        case.assertTrue(np.array_equal(g._c[key], fresh[key]), key)
    case.assertTrue(np.array_equal(g._c["classes"][g._c["cls"]], model.predict(g.X)))


class TestCachedGateExact(unittest.TestCase):
    def setUp(self):
        X, y = problem(3, n=600)
        self.X, self.y = X, y
        self.model = CFMNN(theta=0.15).fit(X, y)
        rng = np.random.default_rng(5)
        # the training rows (many ties: membership 1 in several boxes) and uniform rows
        self.Xg = np.vstack([X, rng.uniform(size=(200, X.shape[1]))])
        self.yg = np.concatenate([y, (self.model.predict(self.Xg[len(X):]))])

    def test_cached_predictions_equal_predict_after_shrinks_and_removals(self):
        rng = np.random.default_rng(1)
        m = copy.deepcopy(self.model)
        g = CachedGate(self.Xg, self.yg, 10 ** 6)
        g.measure(m)
        assert_cache_is_exact(self, g, m)
        n_removed = n_moved = 0
        for _ in range(120):
            k = int(rng.integers(m.n_boxes_))
            p = int(rng.choice(m.members_[k]))
            before = m.n_boxes_
            info = shrink_box(m, k, self.X[p], int(rng.integers(self.X.shape[1])), self.X)
            n_removed += info["removed"]
            n_moved += not info["removed"]
            self.assertEqual(m.n_boxes_, before - info["removed"])
            self.assertEqual(g.measure(m), classification_metrics(self.yg, m.predict(self.Xg)))
            self.assertTrue(np.array_equal(g._c["classes"][g._c["cls"]], m.predict(self.Xg)))
            assert_cache_is_exact(self, g, m)
        self.assertGreater(n_removed, 0)    # both kinds of edit were exercised
        self.assertGreater(n_moved, 0)

    def test_unchanged_model_and_a_full_recompute_fallback(self):
        m = copy.deepcopy(self.model)
        g = CachedGate(self.Xg, self.yg, 10 ** 6)
        first = g.measure(m)
        self.assertEqual(g.measure(m), first)            # nothing changed
        self.assertIsNone(g._undo)
        # two boxes change at once, and a box is added: both take the full recompute
        shrink_box(m, 0, self.X[int(m.members_[0][0])], 0, self.X)
        shrink_box(m, 1, self.X[int(m.members_[1][0])], 1, self.X)
        self.assertEqual(g.measure(m), classification_metrics(self.yg, m.predict(self.Xg)))
        assert_cache_is_exact(self, g, m)
        m.add_points(np.array([[0.5] * self.X.shape[1], [0.9] * self.X.shape[1]]), np.array([0, 1]))
        self.assertEqual(g.measure(m), classification_metrics(self.yg, m.predict(self.Xg)))
        assert_cache_is_exact(self, g, m)

    def test_a_rejected_edit_restores_the_cache(self):
        rng = np.random.default_rng(2)
        m = copy.deepcopy(self.model)
        # a threshold below zero rejects every edit that reaches the metric check
        g = CachedGate(self.Xg, self.yg, 10 ** 6, max_f1_drop=-1000.0)
        before = g.measure(m)
        n_removed = n_moved = 0
        for _ in range(60):
            snap = copy.deepcopy(m)
            k = int(rng.integers(m.n_boxes_))
            p = int(rng.choice(m.members_[k]))
            info = shrink_box(m, k, self.X[p], int(rng.integers(self.X.shape[1])), self.X)
            n_removed += info["removed"]
            n_moved += not info["removed"]
            ok, reason, _ = g.check(before, m)
            self.assertEqual((ok, reason), (False, "macro-F1"))
            m = snap
            assert_cache_is_exact(self, g, m)
            self.assertEqual(g.measure(m), before)
        self.assertGreater(n_removed, 0)
        self.assertGreater(n_moved, 0)

    def test_an_accepted_edit_keeps_the_new_cache(self):
        m = copy.deepcopy(self.model)
        g = CachedGate(self.Xg, self.yg, 10 ** 6, max_f1_drop=1000.0, max_recall_drop=1000.0)
        before = g.measure(m)
        shrink_box(m, 2, self.X[int(m.members_[2][0])], 0, self.X)
        ok, reason, after = g.check(before, m)
        self.assertEqual((ok, reason), (True, ""))
        self.assertEqual(after, classification_metrics(self.yg, m.predict(self.Xg)))
        assert_cache_is_exact(self, g, m)

    def test_box_cap_rejection_matches_the_standard_gate(self):
        m = copy.deepcopy(self.model)
        cap = m.n_boxes_ - 1
        fast, std = CachedGate(self.Xg, self.yg, cap), Gate(self.Xg, self.yg, cap)
        before = fast.measure(m)
        self.assertEqual(before, std.measure(m))
        self.assertEqual(fast.check(before, m), std.check(before, m))
        self.assertEqual(fast.check(before, m), (False, "box cap", None))
        assert_cache_is_exact(self, fast, m)

    def test_decisions_equal_the_standard_gate_edit_by_edit(self):
        rng = np.random.default_rng(4)
        m = copy.deepcopy(self.model)
        fast = CachedGate(self.Xg, self.yg, int(1.1 * m.n_boxes_))
        std = Gate(self.Xg, self.yg, int(1.1 * m.n_boxes_))
        bf, bs = fast.measure(m), std.measure(m)
        self.assertEqual(bf, bs)
        verdicts = set()
        for _ in range(80):
            snap = copy.deepcopy(m)
            k = int(rng.integers(m.n_boxes_))
            p = int(rng.choice(m.members_[k]))
            shrink_box(m, k, self.X[p], int(rng.integers(self.X.shape[1])), self.X)
            rf, rs = fast.check(bf, m), std.check(bs, m)
            self.assertEqual(rf, rs)
            verdicts.add(rf[:2])
            if rf[0]:
                bf, bs = rf[2], rs[2]
            else:
                m = snap
        self.assertGreaterEqual(len(verdicts), 2)         # accepted and rejected edits both occurred


class TestSelfRepairWithFastGate(unittest.TestCase):
    def setUp(self):
        X, y = problem(8, n=1500)
        roles = data.role_split(y, 8)
        self.Xtr, self.ytr = X[roles["train"]], y[roles["train"]]
        self.Xa, self.ya = X[roles["accept"]], y[roles["accept"]]
        self.m = CFMNN(theta=0.2).fit(self.Xtr, self.ytr)

    def both(self, **kw):
        std = run_self_repair(self.m, self.Xtr, self.ytr, self.Xtr, self.ytr, budget=80, n_rounds=4,
                              seed=1, gate_labels_are_new=False, **kw)
        fast = run_self_repair(self.m, self.Xtr, self.ytr, self.Xtr, self.ytr, budget=80, n_rounds=4,
                               seed=1, gate_labels_are_new=False, fast_gate=True, **kw)
        return std, fast

    def check_same(self, std, fast):
        self.assertTrue(same_model(std[0], fast[0]))
        self.assertEqual(std[2], fast[2])                  # the whole per-edit log
        self.assertEqual(std[1], fast[1])                  # and the per-round log
        validate_structure(fast[0], self.Xtr, self.ytr)

    def test_default_gate(self):
        std, fast = self.both()
        self.check_same(std, fast)
        self.assertGreater(sum(e["accepted"] for e in std[2]), 0)

    def test_strict_gate(self):
        std, fast = self.both(max_f1_drop=0.0, max_recall_drop=0.0)
        self.check_same(std, fast)

    def test_loose_gate(self):
        std, fast = self.both(max_f1_drop=1.0, max_recall_drop=2.0)
        self.check_same(std, fast)
        self.assertGreater(sum(e["accepted"] for e in std[2]), 0)

    def test_no_gate(self):
        inf = float("inf")
        std, fast = self.both(max_f1_drop=inf, max_recall_drop=inf)
        self.check_same(std, fast)

    def test_gate_on_other_rows_and_cumulative_reference(self):
        for kw in ({}, {"gate_reference": "start"}):
            std = run_self_repair(self.m, self.Xtr, self.ytr, self.Xa, self.ya, budget=60, n_rounds=3,
                                  seed=2, **kw)
            fast = run_self_repair(self.m, self.Xtr, self.ytr, self.Xa, self.ya, budget=60, n_rounds=3,
                                   seed=2, fast_gate=True, **kw)
            self.check_same(std, fast)

    def test_the_cached_gate_reports_the_same_accepted_edits_to_the_hook(self):
        calls = {False: [], True: []}
        for fg in (False, True):
            run_self_repair(self.m, self.Xtr, self.ytr, self.Xtr, self.ytr, budget=40, n_rounds=2, seed=3,
                            gate_labels_are_new=False, fast_gate=fg,
                            on_accept=lambda b, a, r, fg=fg: calls[fg].append((b.n_boxes_, a.n_boxes_, r)))
        self.assertEqual(calls[False], calls[True])


class TestDefaultsAndReference(unittest.TestCase):
    def setUp(self):
        X, y = problem(8, n=1500)
        roles = data.role_split(y, 8)
        self.Xtr, self.ytr = X[roles["train"]], y[roles["train"]]
        self.m = CFMNN(theta=0.2).fit(self.Xtr, self.ytr)

    def run_sr(self, **kw):
        return run_self_repair(self.m, self.Xtr, self.ytr, self.Xtr, self.ytr, budget=100, n_rounds=4,
                               seed=1, gate_labels_are_new=False, **kw)

    def test_defaults_are_unchanged(self):
        sig = inspect.signature(run_self_repair).parameters
        self.assertIs(sig["fast_gate"].default, False)
        self.assertEqual(sig["gate_reference"].default, "last")
        a, b = self.run_sr(), self.run_sr(fast_gate=False, gate_reference="last")
        self.assertTrue(same_model(a[0], b[0]))
        self.assertEqual(a[2], b[2])

    def test_unknown_reference_raises(self):
        with self.assertRaises(ValueError):
            self.run_sr(gate_reference="first")

    def test_start_reference_bounds_the_total_loss(self):
        f1_drop, rec_drop = 0.2, 0.4
        start = classification_metrics(self.ytr, self.m.predict(self.Xtr))
        for fast in (False, True):
            out, _, edits = self.run_sr(max_f1_drop=f1_drop, max_recall_drop=rec_drop,
                                        gate_reference="start", fast_gate=fast)
            end = classification_metrics(self.ytr, out.predict(self.Xtr))
            self.assertLessEqual(start["macro_f1"] - end["macro_f1"], f1_drop + 1e-9)
            for key, val in start.items():
                if key.startswith("recall_"):
                    self.assertLessEqual(val - end[key], rec_drop + 1e-9, key)
            self.assertGreater(sum(e["accepted"] for e in edits), 0)

    def test_start_and_last_references_differ_only_when_the_loss_builds_up(self):
        # with no limit, both references accept everything, so they must agree
        inf = float("inf")
        a = self.run_sr(max_f1_drop=inf, max_recall_drop=inf)
        b = self.run_sr(max_f1_drop=inf, max_recall_drop=inf, gate_reference="start")
        self.assertTrue(same_model(a[0], b[0]))


if __name__ == "__main__":
    unittest.main()
