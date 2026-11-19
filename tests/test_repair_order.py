"""Run H: the order in which self-repair tries training errors (random, box, greedy)."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_repair_order as ro  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.active import ORDERS, _trial_copy, box_order, run_self_repair  # noqa: E402
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.refine import CHOOSERS, CachedGate, Gate, shrink_box, shrink_target  # noqa: E402

ARRAYS = ("V_", "W_", "sum_", "count_", "box_class_", "centroid_")


def problem(seed=0, n=600, d=5, classes=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    y = ((X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.15, n)) > 0.75).astype(int)
    if classes == 3:
        y = y + (X[:, 2] > 0.8)
    return X, y


def same_model(a, b):
    return (all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ARRAYS)
            and len(a.members_) == len(b.members_)
            and all(np.array_equal(p, q) for p, q in zip(a.members_, b.members_)))


def errors_and_perm(model, X, y, seed):
    t = np.searchsorted(model.classes_, y)
    errors = np.flatnonzero(model.predict_details(X).cls != t)
    return errors, np.random.default_rng(seed).permutation(errors)


def repair(model, X, y, **kw):
    args = dict(chooser="volume", budget=60, n_rounds=4, seed=7, gate_labels_are_new=False)
    args.update(kw)
    return run_self_repair(model, X, y, X, y, **args)


class Toy(unittest.TestCase):
    def setUp(self):
        self.X, self.y = problem(3, d=3)          # 3 features: overlapping boxes, many ties
        self.model = CFMNN().fit(self.X, self.y)
        self.errors, self.perm = errors_and_perm(self.model, self.X, self.y, 7)
        self.assertGreater(len(self.errors), 20)


class TestRandomOrderUnchanged(Toy):
    def test_default_is_the_random_order(self):
        a, _, ea = repair(self.model, self.X, self.y)
        b, _, eb = repair(self.model, self.X, self.y, order="random")
        self.assertTrue(same_model(a, b))
        self.assertEqual(ea, eb)
        self.assertEqual([e["train_index"] for e in ea], [int(p) for p in self.perm[:60]])

    def test_reproduces_the_committed_rq3_edits(self):
        import check_fast_gate as cf
        try:
            rows = cf.run_one("thyroid", 51)
        except (FileNotFoundError, OSError) as err:      # data or committed results not here
            self.skipTest(str(err))
        row = rows[0] if isinstance(rows, list) else rows
        self.assertEqual(row["edit_mismatches"], 0)
        self.assertEqual(row["macro_f1_abs_diff"], 0.0)

    def test_bad_orders_are_refused(self):
        with self.assertRaises(ValueError):
            repair(self.model, self.X, self.y, order="nope")
        with self.assertRaises(ValueError):
            repair(self.model, self.X, self.y, order="greedy", chooser="random")
        self.assertEqual(ORDERS, ("random", "box", "greedy"))


class TestTrialCopy(Toy):
    def test_a_trial_edit_leaves_the_model_alone_and_matches_a_deep_copy(self):
        rng = np.random.default_rng(2)
        keep = copy.deepcopy(self.model)
        removed = 0
        for _ in range(60):
            k = int(rng.integers(self.model.n_boxes_))
            p = int(rng.choice(self.model.members_[k]))
            i = int(rng.integers(self.X.shape[1]))
            light, deep = _trial_copy(self.model), copy.deepcopy(self.model)
            a = shrink_box(light, k, self.X[p], i, self.X)
            b = shrink_box(deep, k, self.X[p], i, self.X)
            removed += a["removed"]
            self.assertEqual(a, b)
            self.assertTrue(same_model(light, deep))
            self.assertTrue(np.array_equal(light.predict(self.X), deep.predict(self.X)))
            self.assertTrue(same_model(self.model, keep))
        self.assertGreater(removed, 0)


class TestBoxOrder(Toy):
    def test_groups_are_whole_biggest_first_and_keep_the_random_order(self):
        out = box_order(self.model, self.X, self.perm)
        self.assertEqual(sorted(out.tolist()), sorted(self.perm.tolist()))
        win = np.asarray(self.model.predict_details(self.X[out]).winner)
        size = {b: int(np.sum(win == b)) for b in set(win.tolist())}
        sizes = [size[b] for b in win]
        self.assertEqual(sizes, sorted(sizes, reverse=True))           # biggest group first
        pos = {int(p): j for j, p in enumerate(self.perm)}
        seen, first = [], {}
        for j, b in enumerate(win):
            if b not in first:
                first[b] = j
                seen.append(b)
            else:
                self.assertEqual(win[j - 1], b)                          # each group is whole
        for b in seen:                                                   # random order inside
            rows = [pos[int(p)] for p, w in zip(out, win) if w == b]
            self.assertEqual(rows, sorted(rows))
        for a, b in zip(seen, seen[1:]):                                 # equal sizes: first seen
            if size[a] == size[b]:
                fa = min(pos[int(p)] for p, w in zip(out, win) if w == a)
                fb = min(pos[int(p)] for p, w in zip(out, win) if w == b)
                self.assertLess(fa, fb)
        self.assertGreater(len(seen), 1)

    def test_box_order_tries_the_same_errors_when_the_budget_covers_them(self):
        _, _, e = repair(self.model, self.X, self.y, order="box", budget=10 ** 6)
        self.assertEqual(sorted(x["train_index"] for x in e), sorted(self.perm.tolist()))


class TestPeek(Toy):
    def test_peek_predicts_a_trial_and_leaves_the_cache(self):
        for kind in (Gate, CachedGate):
            g = kind(self.X, self.y, 10 ** 6)
            m = copy.deepcopy(self.model)
            first = g.measure(m)
            rng = np.random.default_rng(4)
            for _ in range(40):
                k = int(rng.integers(m.n_boxes_))
                p = int(rng.choice(m.members_[k]))
                trial = _trial_copy(m)
                shrink_box(trial, k, self.X[p], int(rng.integers(self.X.shape[1])), self.X)
                cls, met = g.peek(trial)
                pred = trial.predict(self.X)
                self.assertTrue(np.array_equal(trial.classes_[cls], pred))
                self.assertEqual(met, classification_metrics(self.y, pred))
                self.assertEqual(g.measure(m), first)                  # cache still for m
                cls0, _ = g.peek(m)
                self.assertTrue(np.array_equal(m.classes_[cls0], m.predict(self.X)))
            if kind is CachedGate:
                fresh = g._full(m)
                for key in ("M", "D", "top", "cls"):
                    self.assertTrue(np.array_equal(g._c[key], fresh[key]), key)


class TestGreedy(Toy):
    def brute_key(self, model, p, gate, before):
        """Greedy's key for error p, by brute force (deep copy, full predictions)."""
        x, t = self.X[p], int(np.searchsorted(model.classes_, self.y[p]))
        det = model.predict_details(x[None])
        if int(det.cls[0]) == t:
            return (2, 0)
        k = shrink_target(model, x, t, det)
        if k < 0:
            return (2, 0)
        m = copy.deepcopy(model)
        i = CHOOSERS["volume"](m, x, k, int(det.cls[0]), t, None, X_lab=self.X)
        shrink_box(m, k, x, i, self.X)
        gain = int(np.sum(m.predict(self.X) == self.y) - np.sum(model.predict(self.X) == self.y))
        ok = m.n_boxes_ <= gate.box_cap and gate.verdict(before, gate.measure(m))[0]
        return (0 if ok else 1, -gain)

    def test_the_first_try_is_the_best_scored_error(self):
        _, _, e = repair(self.model, self.X, self.y, order="greedy")
        gate = Gate(self.X, self.y, int(np.floor(1.1 * self.model.n_boxes_)))
        before = gate.measure(self.model)
        keys = {int(p): (self.brute_key(self.model, int(p), gate, before), j)
                for j, p in enumerate(self.perm)}
        best = min(keys, key=keys.get)
        self.assertEqual(e[0]["train_index"], best)
        self.assertEqual((e[0]["greedy_class"], -e[0]["greedy_gain"]), keys[best][0])

    def test_scores_equal_the_real_fixes_minus_breaks(self):
        seen = []

        def hook(before, after, removed):
            seen.append(int(np.sum(after.predict(self.X) == self.y)
                            - np.sum(before.predict(self.X) == self.y)))

        _, _, e = repair(self.model, self.X, self.y, order="greedy", fast_gate=True, on_accept=hook)
        accepted = [x for x in e if x["accepted"]]
        self.assertGreater(len(accepted), 3)
        self.assertEqual([int(x["greedy_gain"]) for x in accepted], seen)
        for x in e:                                # a pass predicted for every edit that passed
            if x["edit"] == "shrink":
                self.assertEqual(x["accepted"], x["greedy_class"] == 0)
            else:
                self.assertEqual(x["greedy_class"], 2)

    def test_cached_and_standard_gates_agree(self):
        a, ra, ea = repair(self.model, self.X, self.y, order="greedy", fast_gate=True)
        b, rb, eb = repair(self.model, self.X, self.y, order="greedy", fast_gate=False)
        self.assertTrue(same_model(a, b))
        self.assertEqual(len(ea), len(eb))
        for x, z in zip(ea, eb):
            self.assertEqual({k: v for k, v in x.items() if k != "greedy_gain"},
                             {k: v for k, v in z.items() if k != "greedy_gain"})
        self.assertEqual(ra[-1]["trial_edits"], rb[-1]["trial_edits"])

    def test_budget_rounds_and_each_error_once(self):
        _, rounds, e = repair(self.model, self.X, self.y, order="greedy", budget=25)
        idx = [x["train_index"] for x in e]
        self.assertEqual(len(idx), 25)
        self.assertEqual(len(set(idx)), 25)
        self.assertTrue(set(idx) <= set(self.errors.tolist()))
        self.assertEqual([r["train_errors_tried"] for r in rounds], [0, 7, 13, 19, 25])
        trials = [r["trial_edits"] for r in rounds]
        self.assertEqual(trials, sorted(trials))
        self.assertGreater(trials[-1], 0)
        _, _, e_all = repair(self.model, self.X, self.y, order="greedy", budget=10 ** 6)
        self.assertEqual(sorted(x["train_index"] for x in e_all), sorted(self.perm.tolist()))


class TestRealTries(Toy):
    """Amendment 1: count_skips=False counts only the tries that reach the gate."""

    def test_default_counts_skips(self):
        a, _, ea = repair(self.model, self.X, self.y, budget=40)
        b, _, eb = repair(self.model, self.X, self.y, budget=40, count_skips=True)
        self.assertTrue(same_model(a, b))
        self.assertEqual(ea, eb)
        self.assertEqual(len(ea), 40)

    def test_real_tries_gate_the_budget_and_follow_the_random_order(self):
        budget = 15
        _, rounds, e = repair(self.model, self.X, self.y, budget=budget, count_skips=False)
        gated = [x for x in e if x["edit"] == "shrink"]
        idx = [x["train_index"] for x in e]
        self.assertEqual(idx, [int(p) for p in self.perm[:len(idx)]])    # a prefix of the order
        if len(idx) < len(self.perm):                                    # did not run out
            self.assertEqual(len(gated), budget)
            self.assertEqual(e[-1]["edit"], "shrink")                    # stops on the last gated try
        else:
            self.assertLessEqual(len(gated), budget)
        self.assertEqual(len(rounds), 5)                                 # n_rounds=4, plus round 0
        tried = [r["train_errors_tried"] for r in rounds]
        self.assertEqual(tried, sorted(tried))
        self.assertEqual(tried[-1], len(e))
        per_round = [sum(1 for x in gated if x["round"] == r) for r in range(1, 5)]
        quotas = [len(c) for c in np.array_split(np.arange(min(budget, len(self.perm))), 4)]
        if len(idx) < len(self.perm):
            self.assertEqual(per_round, quotas)

    def test_real_tries_make_the_same_edits_as_the_long_random_run(self):
        # Same order, same model after each edit: the real-tries run is a prefix of trying all.
        _, _, e_real = repair(self.model, self.X, self.y, budget=15, count_skips=False)
        _, _, e_all = repair(self.model, self.X, self.y, budget=10 ** 6)
        strip = lambda es: [{k: v for k, v in x.items() if k != "round"} for x in es]
        self.assertEqual(strip(e_real), strip(e_all[:len(e_real)]))
        m_real, _, _ = repair(self.model, self.X, self.y, budget=10 ** 6, count_skips=False)
        m_all, _, _ = repair(self.model, self.X, self.y, budget=10 ** 6)
        self.assertTrue(same_model(m_real, m_all))                      # no budget limit: identical

    def test_greedy_refuses_real_tries(self):
        with self.assertRaises(ValueError):
            repair(self.model, self.X, self.y, order="greedy", count_skips=False)


class TestRunH(unittest.TestCase):
    def part(self):
        X, y = problem(9, n=1100, d=3)
        return {"train": (X[:500], y[:500]), "pool": (X[500:900], y[500:900]),
                "test": (X[900:], y[900:])}

    def test_arms_are_built_as_specified(self):
        calls = []
        real = ro.run_self_repair

        def spy(*a, **kw):
            calls.append(kw)
            return real(*a, **kw)

        with mock.patch.object(ro, "split", lambda name, seed: self.part()), \
                mock.patch.object(ro, "run_self_repair", spy), \
                mock.patch.object(ro, "use_fast_gate", lambda: True):
            rows = ro.run_one("toy", 131)
        self.assertEqual([r["arm"] for r in rows], ro.MODELS)
        self.assertEqual(len(calls), 8)
        by = dict(zip(ro.MODELS[1:], calls))
        d = min(300, 400 // 4)
        self.assertEqual({a: (kw["order"], kw["budget"], kw.get("count_skips", True))
                          for a, kw in by.items()},
                         {"random-300": ("random", d, True), "random-1000": ("random", 1000, True),
                          "random-all": ("random", 500, True),
                          "random-300-real": ("random", d, False),
                          "box-300": ("box", d, True), "box-1000": ("box", 1000, True),
                          "greedy-300": ("greedy", d, True), "greedy-1000": ("greedy", 1000, True)})
        real = next(r for r in rows if r["arm"] == "random-300-real")
        self.assertLessEqual(real["edits_gated"], d)
        for r in rows[1:]:                         # every logged try is gated or skipped
            self.assertEqual(r["edits_logged"],
                             r["edits_gated"] + r["skip_already_right"] + r["skip_not_inside"])
        for kw in calls:                           # Run A's call, the same seed and model
            self.assertEqual((kw["chooser"], kw["n_rounds"], kw["seed"], kw["gate_labels_are_new"],
                              kw["fast_gate"], kw.get("max_f1_drop", 0.5),
                              kw.get("max_recall_drop", 1.0), kw.get("gate_reference", "last")),
                             ("volume", 10, 131, False, True, 0.5, 1.0, "last"))
        self.assertTrue(all(set(r) == set(ro.COLUMNS) for r in rows))
        self.assertEqual({r["boxes_start"] for r in rows}, {rows[0]["boxes_start"]})
        self.assertEqual(rows[0]["churn_changed"], 0)
        greedy = [r for r in rows if r["order"] == "greedy"]
        self.assertTrue(all(r["trial_edits"] > 0 for r in greedy))
        self.assertTrue(all(r["trial_edits"] == 0 for r in rows[1:] if r["order"] != "greedy"))

    def test_refuses_without_run_e_pass(self):
        with mock.patch.object(ro, "split", lambda name, seed: self.part()), \
                mock.patch.object(ro, "use_fast_gate", lambda: False):
            with self.assertRaises(RuntimeError):
                ro.run_one("toy", 131)

    def table(self, shift=0.0):
        recs = []
        for s in ro.SEEDS:
            for arm in ro.MODELS:
                rec = {c: 0.0 for c in ro.COLUMNS}
                rec.update({"dataset": "thyroid", "seed": s, "arm": arm, "order": arm.split("-")[0],
                            "gate_impl": "cached"})
                rec["test_macro_f1"] = 50.0 + 0.01 * s + (shift * (s - 129) if arm == "greedy-300" else 0)
                recs.append(rec)
        return pd.DataFrame(recs)

    def test_primary_verdict_and_summary(self):
        res = ro.comparisons(self.table(0.5))
        self.assertEqual(list(res)[0], ("greedy-300", "random-300"))
        self.assertEqual(res[("greedy-300", "random-300")].verdict.tolist(), ["better"])
        self.assertEqual(res[("box-300", "random-300")].verdict.tolist(), ["level"])
        self.assertEqual(ro.comparisons(self.table(-0.5))[("greedy-300", "random-300")]
                         .verdict.tolist(), ["worse"])
        with tempfile.TemporaryDirectory() as tmp:
            text = ro.summarise(self.table(0.5), Path(tmp))
            self.assertTrue((Path(tmp) / "summary.md").exists())
        self.assertIn("## Primary: greedy-300 vs random-300, test macro-F1", text)
        self.assertIn("**greedy-300 vs random-300**: better 1, level 0, worse 0", text)

    def test_amendment_comparisons_and_reading(self):
        t = self.table(0.5)                        # greedy-300 better on macro-F1 than both randoms
        res = ro.comparisons(t)
        self.assertEqual(res[("greedy-300", "random-300-real")].verdict.tolist(), ["better"])
        t.loc[t.arm == "random-300-real", "test_macro_f1"] = t.loc[t.arm == "greedy-300",
                                                                   "test_macro_f1"].to_numpy()
        self.assertEqual(ro.comparisons(t)[("greedy-300", "random-300-real")].verdict.tolist(),
                         ["level"])
        self.assertEqual(ro.reading("better", "better"), "a better order")
        self.assertEqual(ro.reading("better", "level"), "the gain comes from not wasting tries")
        self.assertEqual(ro.reading("level", "worse"), "level (against real tries: worse)")
        g = (t.arm == "greedy-300").to_numpy()
        t["seconds"] = np.where(g, 100.0 + t.seed, 1.0)            # greedy slower every seed
        t["edits_gated"] = np.where(g, 300.0, 1000.0 + t.seed)    # but gates fewer edits
        eff = ro.efficiency(t)
        self.assertEqual(eff[("greedy-300", "random-all", "seconds")].verdict.tolist(), ["slower"])
        self.assertEqual(eff[("greedy-300", "random-all", "edits_gated")].verdict.tolist(), ["fewer"])
        with tempfile.TemporaryDirectory() as tmp:
            text = ro.summarise(t, Path(tmp))
        self.assertIn("## Primary: greedy-300 vs random-300, test macro-F1", text)
        self.assertIn("## Amendment 1: greedy-300 vs random-300-real, test macro-F1", text)
        self.assertIn("| thyroid | better | level | the gain comes from not wasting tries |", text)
        self.assertIn("**greedy-300 vs random-all** (seconds): slower 1", text)
        self.assertIn("**greedy-300 vs random-all** (edits_gated): fewer 1", text)

    def test_study_settings(self):
        self.assertEqual(ro.SEEDS, list(range(130, 140)))
        self.assertEqual(ro.DATASETS, ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"])
        self.assertEqual(ro.PRIMARY, ("greedy-300", "random-300"))
        self.assertEqual(ro.AMENDED, [("greedy-300", "random-300-real")])
        self.assertEqual(len(ro.MODELS), 9)       # 6 datasets x 10 seeds x 9 rows = 540


if __name__ == "__main__":
    unittest.main()
