"""Tests for selective refinement (blueprint 7, 11.2: cfmnn_core.add_points, refine, active)."""

import copy
import unittest

import numpy as np

from pace_cfmnn import CFMNN, data
from pace_cfmnn.active import (Context, LabelOracle, _oracle_cut, run_refinement, run_self_repair,
                               select_knn, select_oracle_errors, top_b)
from pace_cfmnn.audit import impact_set
from pace_cfmnn.explain import PaceShap, contrastive_pair
from pace_cfmnn.neighborhood import Perturber
from pace_cfmnn.refine import (Gate, contains, face_cut, feature_explained, feature_fewest_members,
                               feature_pace_shap, feature_random, feature_volume, members_cut,
                               shrink_box, shrink_target, validate_structure)
from pace_cfmnn.signals import SignalComputer

ARRAYS = ("V_", "W_", "sum_", "count_", "box_class_", "centroid_")


def problem(seed=0, n=700, d=5, classes=3):
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


def overlap_case():
    """A class-0 box [0.1, 0.5] x [0.1, 0.9] reaching into class 1's strip near x0 = 0.47."""
    X = np.array([[0.1, 0.5], [0.8, 0.5], [0.5, 0.5], [0.9, 0.5], [0.3, 0.1], [0.3, 0.9]])
    y = np.array([0, 1, 0, 1, 0, 0])
    m = CFMNN(theta=0.65).fit(X, y)   # size limit 1.3 > 1.2, clear of rounding
    X_acc = np.array([[0.47, 0.5], [0.48, 0.5], [0.2, 0.5], [0.3, 0.3]])
    y_acc = np.array([1, 1, 0, 0])
    return m, X, y, np.array([0.46, 0.45]), X_acc, y_acc


def tie_case():
    """overlap_case's rows plus x itself as a class-1 training row.

    x lies inside both boxes (membership 1 in each), and the centroid rule
    gives it to the class-0 box: a training error that is a tie, as on the
    real datasets.
    """
    m, X, y, x, X_acc, y_acc = overlap_case()
    X, y = np.vstack([X, x]), np.append(y, 1)
    return CFMNN(theta=0.65).fit(X, y), X, y, X_acc, y_acc


class TestAddPoints(unittest.TestCase):
    def test_continuing_equals_one_fit(self):
        for seed in range(3):
            X, y = problem(seed)
            for rule in ("fixed", "original"):
                for expansion in ("search", "best"):
                    full = CFMNN(centroid_rule=rule, expansion=expansion).fit(X, y)
                    part = CFMNN(centroid_rule=rule, expansion=expansion).fit(X[:400], y[:400])
                    part.add_points(X[400:], y[400:], rows=np.arange(400, len(X)))
                    self.assertTrue(same_model(full, part), (seed, rule, expansion))
                    for p, q in zip(full.class_boxes_, part.class_boxes_):
                        np.testing.assert_array_equal(p, q)

    def test_default_rows_continue_and_unseen_class_refused(self):
        X, y = problem(1, classes=2)
        m = CFMNN().fit(X[:300], y[:300])
        m.add_points(X[300:301], y[300:301])
        self.assertIn(300, np.concatenate(m.members_))
        with self.assertRaises(ValueError):
            m.add_points(X[:1], [7])

    def test_remove_box_keeps_the_rest_in_order(self):
        X, y = problem(2)
        m = CFMNN().fit(X, y)
        V, H = m.V_.copy(), m.n_boxes_
        m._remove_box(3)
        self.assertEqual(m.n_boxes_, H - 1)
        np.testing.assert_array_equal(m.V_, np.delete(V, 3, axis=0))
        for c, idx in enumerate(m.class_boxes_):
            np.testing.assert_array_equal(idx, np.flatnonzero(m.box_class_ == c))


class TestShrink(unittest.TestCase):
    def test_overlap_case_is_fixed_by_the_explained_shrink(self):
        m, X, y, x, X_acc, y_acc = overlap_case()
        self.assertEqual(m.n_boxes_, 2)
        det = m.predict_details(x[None])
        self.assertEqual(det.cls[0], 0)                     # misclassified, true class 1
        k = shrink_target(m, x, 1, det)
        self.assertEqual(k, int(det.winner[0]))
        i = feature_explained(m, x, k, 0, 1)
        e = contrastive_pair(m, x[None], np.array([0]), np.array([1])).e[0]
        self.assertEqual(i, int(np.argmax(e)))
        self.assertEqual(i, 0)                              # x is far from class 1 only on x0
        self.assertEqual(feature_volume(m, x, k, 0, 1), 0)  # the thin cut is also on x0

        gate = Gate(X_acc, y_acc, box_cap=3)
        before = gate.measure(m)
        info = shrink_box(m, k, x, i, X)
        self.assertFalse(contains(m, k, x))
        self.assertAlmostEqual(m.W_[k, 0], 0.459)
        self.assertEqual(info["n_cut"], 1)                  # training row (0.5, 0.5) is cut out
        m.add_points(x, [1], rows=[len(X)])
        X_lab, y_lab = np.vstack([X, x]), np.append(y, 1)
        validate_structure(m, X_lab, y_lab)
        np.testing.assert_allclose(m.sum_[k], X[m.members_[k]].sum(axis=0))
        ok, reason, after = gate.check(before, m)
        self.assertTrue(ok, reason)
        self.assertGreater(after["macro_f1"], before["macro_f1"])
        np.testing.assert_array_equal(m.predict(X_acc), y_acc)

    def test_cut_that_empties_a_box_removes_it(self):
        X = np.array([[0.4, 0.4], [0.9, 0.9]])
        m = CFMNN().fit(X, np.array([0, 1]))
        info = shrink_box(m, 0, np.array([0.4, 0.4]), 1, X)
        self.assertTrue(info["removed"])
        self.assertEqual(m.n_boxes_, 1)
        self.assertEqual(m.box_class_.tolist(), [1])

    def test_face_cut_and_choosers(self):
        m, X, y, x, _, _ = overlap_case()
        self.assertEqual(face_cut(m, 0, x, 0)[0], "high")    # x0 = 0.46 is nearer w = 0.5
        side, new, share = face_cut(m, 0, x, 1)
        self.assertEqual((side, round(new, 3)), ("low", 0.451))         # x1 = 0.45 is nearer v = 0.1
        self.assertAlmostEqual(share, (0.451 - 0.1) / 0.8)
        rng = np.random.default_rng(0)
        self.assertTrue(all(0 <= feature_random(m, x, 0, 0, 1, rng) < 2 for _ in range(20)))

    def test_paper_rule_is_refused(self):
        X, y = problem(3, classes=2)
        m = CFMNN(centroid_rule="original").fit(X, y)
        with self.assertRaises(ValueError):
            shrink_box(m, 0, X[0], 0, X)


class TestGate(unittest.TestCase):
    def test_thresholds(self):
        X, y = problem(4)
        m = CFMNN().fit(X[:500], y[:500])
        gate = Gate(X[500:], y[500:], box_cap=m.n_boxes_)
        before = gate.measure(m)
        self.assertTrue(gate.check(before, m)[0])
        worse = dict(before, macro_f1=before["macro_f1"] + 0.6)
        self.assertEqual(gate.check(worse, m)[1], "macro-F1")
        key = next(k for k in before if k.startswith("recall_"))
        self.assertEqual(gate.check(dict(before, **{key: before[key] + 1.1}), m)[1], "class recall")
        self.assertEqual(Gate(X[500:], y[500:], box_cap=m.n_boxes_ - 1).check(before, m)[1],
                         "box cap")


class TestLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = problem(5, n=2000)
        roles = data.role_split(y, 5)
        tr = np.random.default_rng(5).permutation(roles["train"])
        cls.parts = {r: (X[i], y[i]) for r, i in roles.items()}
        cls.Xtr, cls.ytr = X[tr], y[tr]
        cls.model = CFMNN().fit(cls.Xtr, cls.ytr)
        cls.pert = Perturber(k=10).fit(cls.Xtr)

    def run_arm(self, selector="order_ensemble", operation="A+B+C", seed=0, test=True, **kw):
        P, I, A, T = (self.parts[r] for r in ("pool", "inner", "accept", "test"))
        oracle = LabelOracle(P[1])
        out = run_refinement(self.model, self.Xtr, self.ytr, P[0], oracle, I[0], A[0], A[1],
                             selector, operation, budget=40, n_rounds=4, seed=seed,
                             perturber=self.pert, X_test=T[0] if test else None,
                             y_test=T[1] if test else None, **kw)
        return (*out, oracle)

    def test_oracle_budget_and_input_untouched(self):
        V0 = self.model.V_.copy()
        m, rounds, edits, oracle = self.run_arm()
        self.assertEqual(len(oracle.revealed), 40)
        self.assertEqual(set(oracle.revealed), {e["pool_index"] for e in edits})
        self.assertEqual([r["labels"] for r in rounds], [0, 10, 20, 30, 40])
        np.testing.assert_array_equal(self.model.V_, V0)
        with self.assertRaises(ValueError):
            oracle.reveal(oracle.revealed[:1])

    def test_test_rows_never_change_a_decision(self):
        a = self.run_arm(test=True)
        b = self.run_arm(test=False)
        self.assertTrue(same_model(a[0], b[0]))
        self.assertEqual(a[2], b[2])
        self.assertNotIn("test_macro_f1", b[1][0])

    def test_same_seed_same_run_and_every_selector_runs(self):
        for sel in ("random", "margin", "pd_u_star", "order_ensemble"):
            a, b = self.run_arm(sel, "A+B"), self.run_arm(sel, "A+B")
            self.assertTrue(same_model(a[0], b[0]), sel)
            self.assertEqual(a[2], b[2], sel)

    def test_box_cap_and_rollback(self):
        m, _, edits, _ = self.run_arm(operation="A+B", box_cap_factor=1.0)
        self.assertLessEqual(m.n_boxes_, self.model.n_boxes_)
        accepted = [e for e in edits if e["accepted"]]
        self.assertTrue(any(e["reason"] == "box cap" for e in edits))
        n_members = sum(len(mm) for mm in m.members_)
        self.assertEqual(n_members, len(self.Xtr) + len(accepted))   # rejected edits left no trace

    def test_shrink_arms_fall_back_to_add(self):
        for op in ("A+B+C", "A+B+C'-random", "A+B+C'-volume"):
            _, _, edits, _ = self.run_arm(operation=op)
            per_point = {}
            for e in edits:
                per_point.setdefault(e["pool_index"], []).append(e)
            for tries in per_point.values():
                self.assertLessEqual(len(tries), 2)
                if tries[0]["edit"] == "shrink" and not tries[0]["accepted"]:
                    self.assertEqual(tries[1]["edit"], "add")
                self.assertLessEqual(sum(t["accepted"] for t in tries), 1)

    def test_top_b_breaks_ties_at_random(self):
        rng = np.random.default_rng(0)
        picks = {tuple(sorted(top_b(np.zeros(10), 3, rng))) for _ in range(20)}
        self.assertGreater(len(picks), 1)
        np.testing.assert_array_equal(top_b(np.arange(10.0), 3, rng), [9, 8, 7])

    def test_label_columns(self):
        _, rounds, _, _ = self.run_arm(operation="A+B")
        n_gate = len(self.parts["accept"][1])
        for r in rounds:
            self.assertEqual(r["pool_labels"], r["labels"])
            self.assertEqual(r["gate_labels"], n_gate)
            self.assertEqual(r["total_labels"], r["labels"] + n_gate)

    def test_v2_arms_run_and_repeat(self):
        bg = self.Xtr[:20]
        for sel, op in (("knn", "A+B+C'-volume"), ("oracle_errors", "A+B+C'-volume"),
                        ("order_ensemble", "A+B+C-fewest"), ("order_ensemble", "A+B+C-oracle"),
                        ("random", "A+B+C-paceshap")):
            a = self.run_arm(sel, op, background=bg)
            b = self.run_arm(sel, op, background=bg)
            self.assertTrue(same_model(a[0], b[0]), (sel, op))
            self.assertEqual(a[2], b[2], (sel, op))
            self.assertEqual(len(a[3].revealed), 40, (sel, op))

    def test_oracle_errors_picks_errors_and_buys_only_its_picks(self):
        _, _, edits, oracle = self.run_arm("oracle_errors", "A+B")
        _, _, rand, _ = self.run_arm("random", "A+B")
        hit = np.mean([e["was_wrong"] for e in edits])
        self.assertGreater(hit, 2 * np.mean([e["was_wrong"] for e in rand]))
        self.assertEqual(len(oracle.revealed), 40)          # peeking bought nothing

    def test_on_accept_sees_every_accepted_edit_and_the_certificate_holds(self):
        T = self.parts["test"][0]
        seen = []

        def hook(before, after, removed):
            changed = before.predict(T) != after.predict(T)
            seen.append(bool((changed & ~impact_set(before, after, T, removed, tight=True)).any()))

        _, _, edits, _ = self.run_arm(operation="A+B+C'-volume", on_accept=hook)
        self.assertEqual(len(seen), sum(e["accepted"] for e in edits))
        self.assertFalse(any(seen))


class TestV2Choosers(unittest.TestCase):
    def test_knn_and_oracle_errors_scores(self):
        X, y = problem(9, n=600)
        m = CFMNN().fit(X[:400], y[:400])
        ctx = Context(m, X[400:], X[:400], y[:400], X[:50], None, np.random.default_rng(0), 0)
        s = select_knn(ctx)
        self.assertEqual(s.shape, (200,))
        self.assertTrue(((s >= 0) & (s <= 1)).all())
        with self.assertRaises(ValueError):
            select_oracle_errors(ctx)
        ctx.y_cand_hidden = y[400:]
        np.testing.assert_array_equal(select_oracle_errors(ctx),
                                      (m.predict(X[400:]) != y[400:]).astype(float))

    def test_fewest_members_counts_match_the_shrink(self):
        X, y = problem(7, n=400, d=3)
        m = CFMNN().fit(X, y)
        rng = np.random.default_rng(0)
        differs = 0
        for _ in range(100):
            k = int(rng.integers(m.n_boxes_))
            x = m.V_[k] + rng.uniform(size=3) * (m.W_[k] - m.V_[k])
            cut = []
            for i in range(3):
                cut.append(shrink_box(copy.deepcopy(m), k, x, i, X)["n_cut"])
                self.assertEqual(members_cut(m, k, x, i, X), cut[-1])
            best = min((cut[i], face_cut(m, k, x, i)[2], i) for i in range(3))[2]
            pick = feature_fewest_members(m, x, k, 0, 1, X_lab=X)
            self.assertEqual(pick, best)
            differs += pick != feature_volume(m, x, k, 0, 1)
        self.assertGreater(differs, 0)                      # not the volume rule in disguise
        with self.assertRaises(ValueError):
            feature_fewest_members(m, x, k, 0, 1)

    def test_pace_shap_takes_the_largest_attribution(self):
        m, X, y, x, _, _ = overlap_case()
        phi = PaceShap(m, X, seed=0).explain(x[None])[0]
        self.assertEqual(feature_pace_shap(m, x, 0, 0, 1, background=X), int(np.argmax(phi)))
        with self.assertRaises(ValueError):
            feature_pace_shap(m, x, 0, 0, 1)

    def test_oracle_cut_keeps_the_gates_best_feature(self):
        m, X, y, x, X_acc, y_acc = overlap_case()
        X_lab = np.vstack([X, x])
        gate = Gate(X_acc, y_acc, box_cap=3)
        before = gate.measure(m)
        k = shrink_target(m, x, 1)
        trial, info, after = _oracle_cut(m, k, x, 1, len(X), X_lab, gate, before)
        keys = {}
        for i in range(2):
            t = copy.deepcopy(m)
            inf = shrink_box(t, k, x, i, X_lab)
            t.add_points(x, [1], rows=[len(X)])
            ok, _, a = gate.check(before, t)
            if ok:
                keys[i] = (a["macro_f1"], -inf["share_cut"], -i)
        self.assertEqual(info["feature"], max(keys, key=keys.get))
        self.assertEqual(after, gate.measure(trial))
        self.assertTrue(same_model(m, overlap_case()[0]))    # the snapshot is not changed
        none = _oracle_cut(m, k, x, 1, len(X), X_lab, Gate(X_acc, y_acc, box_cap=0), before)
        self.assertEqual(none, (None, {}, None))


class TestSelfRepair(unittest.TestCase):
    def test_fixes_a_planted_training_tie_without_new_labels(self):
        m, X, y, X_acc, y_acc = tie_case()
        det = m.predict_details(X)
        self.assertEqual(np.flatnonzero(det.cls != y).tolist(), [6])
        self.assertEqual(int(det.n_tied[6]), 2)
        calls = []
        out, rounds, edits = run_self_repair(m, X, y, X_acc, y_acc, budget=10, n_rounds=1,
                                             on_accept=lambda b, a, r: calls.append(r))
        self.assertEqual(out.predict(X[6:7])[0], 1)
        self.assertEqual([(e["accepted"], e["shrink_feature"]) for e in edits], [(True, 0)])
        self.assertEqual(calls, [None])
        n_members = sum(len(a) for a in out.members_)
        self.assertEqual(n_members, len(X) - edits[0]["shrink_n_cut"])   # nothing was added
        self.assertLessEqual(out.n_boxes_, m.n_boxes_)
        self.assertEqual([r["pool_labels"] for r in rounds], [0, 0])
        self.assertEqual(rounds[-1]["total_labels"], len(y_acc))
        validate_structure(out, X, y)
        self.assertTrue(same_model(m, tie_case()[0]))        # the input model is not changed
        _, rounds, _ = run_self_repair(m, X, y, X, y, budget=10, n_rounds=1, gate_labels_are_new=False)
        self.assertEqual(rounds[-1]["total_labels"], 0)      # gating on training rows costs nothing

    def test_failed_gate_is_rolled_back(self):
        m, X, y, _, _ = tie_case()
        # (0.47, 0.48) is class 0 and right only while the class-0 box still holds it.
        X_gate, y_gate = np.array([[0.47, 0.48], [0.8, 0.5]]), np.array([0, 1])
        calls = []
        out, _, edits = run_self_repair(m, X, y, X_gate, y_gate, budget=10, n_rounds=1,
                                        on_accept=lambda b, a, r: calls.append(r))
        self.assertEqual([(e["accepted"], e["reason"]) for e in edits], [(False, "macro-F1")])
        self.assertTrue(same_model(out, m))
        self.assertEqual(calls, [])

    def test_runs_on_a_larger_problem(self):
        X, y = problem(8, n=1500)
        roles = data.role_split(y, 8)
        Xtr, ytr = X[roles["train"]], y[roles["train"]]
        Xa, ya = X[roles["accept"]], y[roles["accept"]]
        m = CFMNN().fit(Xtr, ytr)
        out, rounds, edits = run_self_repair(m, Xtr, ytr, Xa, ya, budget=60, n_rounds=3, seed=1)
        a = run_self_repair(m, Xtr, ytr, Xa, ya, budget=60, n_rounds=3, seed=1)
        self.assertTrue(same_model(out, a[0]))
        self.assertEqual(len(edits), min(60, int(np.sum(m.predict(Xtr) != ytr))))
        self.assertGreater(sum(e["accepted"] for e in edits), 0)
        self.assertEqual(rounds[-1]["train_errors_tried"], len(edits))
        self.assertLessEqual(out.n_boxes_, m.n_boxes_)
        validate_structure(out, Xtr, ytr)


class TestSignalsWithoutEI(unittest.TestCase):
    def test_pd_and_u_star_unchanged(self):
        X, y = problem(6, classes=2)
        m = CFMNN().fit(X, y)
        p = Perturber(k=15).fit(X)
        a = SignalComputer(m, p).compute(X[:200], seed=2)
        b = SignalComputer(m, p, ei=False).compute(X[:200], seed=2)
        np.testing.assert_array_equal(a.pd, b.pd)
        np.testing.assert_array_equal(a.u_star, b.u_star)
        self.assertIsNone(b.ei)
        self.assertEqual(b.ei_coverage, 0.0)


if __name__ == "__main__":
    unittest.main()
