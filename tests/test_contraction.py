"""Tests for pace_cfmnn/contraction.py (Simpson's overlap test and contraction on a trained CFMNN)
and scripts/run_contraction.py (Run F). The 2-D cases are worked by hand in the comments."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_contraction as rc  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.contraction import (contract_all, contract_at_errors, contract_pair,  # noqa: E402
                                    contract_pair_gfmm, overlap_case, overlap_test, refresh_box,
                                    rows_in_wrong_class_box, rows_without_box)
from pace_cfmnn.refine import shrink_box, validate_structure  # noqa: E402


def boxes(*pairs):
    """V and W arrays from (lower corner, upper corner) pairs."""
    return (np.array([p[0] for p in pairs], float), np.array([p[1] for p in pairs], float))


def toy(seed=0, n=300, overlap=0.25):
    """Three classes in 2-D whose regions overlap, scaled to [0, 1]."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = np.clip(0.2 + 0.3 * y[:, None] + rng.uniform(-overlap, overlap, (n, 2)), 0, 1)
    return X, y


class TestSimpsonCases(unittest.TestCase):
    """Box j and box k in 2-D; each case is the one on feature 0, which has the smaller overlap."""

    def test_case_1(self):
        # j = [0, 0.5] x [0, 1], k = [0.4, 1] x [0.2, 0.8].
        # Feature 0: 0 < 0.4 < 0.5 < 1, case 1, overlap 0.5 - 0.4 = 0.1.
        # Feature 1: 0 < 0.2 <= 0.8 < 1, case 3, overlap min(0.8 - 0, 1 - 0.2) = 0.8.
        # Contract feature 0: w_j = v_k = (0.5 + 0.4) / 2 = 0.45.
        V, W = boxes(([0, 0], [0.5, 1]), ([0.4, 0.2], [1, 0.8]))
        self.assertEqual(overlap_case(0, 0.5, 0.4, 1)[0], 1)
        i, case, delta = overlap_test(V[0], W[0], V[1], W[1])
        self.assertEqual((i, case), (0, 1))
        self.assertAlmostEqual(delta, 0.1)
        info = contract_pair(V, W, 0, 1)
        self.assertEqual(info["changed"], (0, 1))
        self.assertAlmostEqual(W[0, 0], 0.45)
        self.assertAlmostEqual(V[1, 0], 0.45)
        np.testing.assert_array_equal(V[0], [0, 0])
        np.testing.assert_array_equal(W[1], [1, 0.8])
        self.assertEqual(V[1, 1], 0.2)

    def test_case_2(self):
        # The mirror image: j = [0.4, 1] x [0.2, 0.8], k = [0, 0.5] x [0, 1].
        # Feature 0: 0 < 0.4 < 0.5 < 1 with the roles swapped, case 2, overlap 0.1.
        # Feature 1: 0 < 0.2 <= 0.8 < 1, j inside k, case 4, overlap 0.8.
        # Contract feature 0: v_j = w_k = (0.5 + 0.4) / 2 = 0.45.
        V, W = boxes(([0.4, 0.2], [1, 0.8]), ([0, 0], [0.5, 1]))
        self.assertEqual(overlap_case(0.2, 0.8, 0, 1)[0], 4)
        i, case, delta = overlap_test(V[0], W[0], V[1], W[1])
        self.assertEqual((i, case), (0, 2))
        self.assertAlmostEqual(delta, 0.1)
        contract_pair(V, W, 0, 1)
        self.assertAlmostEqual(V[0, 0], 0.45)
        self.assertAlmostEqual(W[1, 0], 0.45)

    def test_case_3_both_branches(self):
        # k inside j. j = [0, 1] x [0, 1], k = [0.1, 0.3] x [0.2, 0.9].
        # Feature 0: overlap min(0.2 - 0, 1 - 0.1) = 0.2; feature 1: min(0.9, 0.7) = 0.7.
        # w_k - v_j = 0.2 < w_j - v_k = 0.9, so v_j = w_k = 0.2: j becomes [0.2, 1] x [0, 1].
        V, W = boxes(([0, 0], [1, 1]), ([0.1, 0.3], [0.2, 0.9]))
        i, case, delta = overlap_test(V[0], W[0], V[1], W[1])
        self.assertEqual((i, case), (0, 3))
        self.assertAlmostEqual(delta, 0.2)
        info = contract_pair(V, W, 0, 1)
        self.assertEqual(info["changed"], (0,))
        np.testing.assert_allclose(V, [[0.2, 0], [0.1, 0.3]])
        np.testing.assert_allclose(W, [[1, 1], [0.2, 0.9]])
        # k = [0.8, 0.3] x [0.9, 0.9]: w_k - v_j = 0.9 >= w_j - v_k = 0.2, so w_j = v_k = 0.8.
        V, W = boxes(([0, 0], [1, 1]), ([0.8, 0.3], [0.9, 0.9]))
        self.assertEqual(overlap_test(V[0], W[0], V[1], W[1])[:2], (0, 3))
        contract_pair(V, W, 0, 1)
        np.testing.assert_allclose(W[0], [0.8, 1])
        np.testing.assert_allclose(V[0], [0, 0])

    def test_case_4_both_branches(self):
        # j inside k. j = [0.1, 0.3] x [0.2, 0.9], k = [0, 1] x [0, 1].
        # Feature 0: overlap min(0.2 - 0, 1 - 0.1) = 0.2. w_k - v_j = 0.9 >= w_j - v_k = 0.2,
        # so v_k = w_j = 0.2: k becomes [0.2, 1] x [0, 1].
        V, W = boxes(([0.1, 0.3], [0.2, 0.9]), ([0, 0], [1, 1]))
        i, case, delta = overlap_test(V[0], W[0], V[1], W[1])
        self.assertEqual((i, case), (0, 4))
        self.assertAlmostEqual(delta, 0.2)
        info = contract_pair(V, W, 0, 1)
        self.assertEqual(info["changed"], (1,))
        np.testing.assert_allclose(V[1], [0.2, 0])
        # j = [0.8, 0.3] x [0.9, 0.9]: w_k - v_j = 0.2 < w_j - v_k = 0.9, so w_k = v_j = 0.8.
        V, W = boxes(([0.8, 0.3], [0.9, 0.9]), ([0, 0], [1, 1]))
        self.assertEqual(overlap_test(V[0], W[0], V[1], W[1])[:2], (0, 4))
        contract_pair(V, W, 0, 1)
        np.testing.assert_allclose(W[1], [0.8, 1])

    def test_tie_goes_to_the_lowest_feature(self):
        # Both features: case 1 with overlap 0.1.
        V, W = boxes(([0, 0], [0.5, 0.5]), ([0.4, 0.4], [1, 1]))
        self.assertEqual(overlap_test(V[0], W[0], V[1], W[1])[0], 0)

    def test_shared_face_is_not_an_overlap(self):
        # j = [0, 0.5] x [0, 1], k = [0.5, 1] x [0.2, 0.8]: on feature 0 they share the face
        # x = 0.5; no strict case fits, so Simpson's test says no overlap and nothing moves.
        V, W = boxes(([0, 0], [0.5, 1]), ([0.5, 0.2], [1, 0.8]))
        self.assertEqual(overlap_test(V[0], W[0], V[1], W[1]), (-1, 0, float("inf")))
        self.assertIsNone(contract_pair(V, W, 0, 1))
        # Equal intervals on a feature also fit no case.
        self.assertEqual(overlap_case(0.2, 0.6, 0.2, 0.6)[0], 0)
        # Disjoint on one feature: no overlap.
        V, W = boxes(([0, 0], [0.3, 1]), ([0.4, 0.2], [1, 0.8]))
        self.assertIsNone(contract_pair(V, W, 0, 1))

    def test_after_contraction_the_pair_no_longer_overlaps(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            a, b = np.sort(rng.uniform(0, 1, (2, 2, 3)), axis=1)
            V, W = np.array([a[0], b[0]]), np.array([a[1], b[1]])
            if contract_pair(V, W, 0, 1) is not None:
                self.assertTrue((V <= W).all())
                self.assertEqual(overlap_test(V[0], W[0], V[1], W[1])[0], -1)


class TestGfmmRule(unittest.TestCase):
    def test_shared_face_is_resolved_with_a_gap(self):
        # The shared face Simpson's test misses: GFMM's test counts it and opens a gap.
        V, W = boxes(([0, 0], [0.5, 1]), ([0.5, 0.2], [1, 0.8]))
        info = contract_pair_gfmm(V, W, 0, 1)
        self.assertIsNotNone(info)
        self.assertEqual(info["feature"], 0)
        self.assertTrue((np.maximum(V[0], V[1]) > np.minimum(W[0], W[1])).any())

    def test_partial_overlap_matches_simpson_up_to_the_gap(self):
        V, W = boxes(([0, 0], [0.5, 1]), ([0.4, 0.2], [1, 0.8]))
        contract_pair_gfmm(V, W, 0, 1)
        self.assertAlmostEqual(W[0, 0], 0.45)
        self.assertAlmostEqual(V[1, 0], 0.45 + 1e-5)

    def test_apart_boxes_are_left_alone(self):
        V, W = boxes(([0, 0], [0.3, 1]), ([0.4, 0.2], [1, 0.8]))
        self.assertIsNone(contract_pair_gfmm(V, W, 0, 1))


class TestOnAModel(unittest.TestCase):
    def setUp(self):
        self.X, self.y = toy()
        self.m = CFMNN(theta=0.3, gamma=4).fit(self.X, self.y)

    def test_refresh_matches_shrink_box(self):
        m = self.m
        k = int(np.argmax(m.count_))
        x = m.centroid_[k]
        a = __import__("copy").deepcopy(m)
        shrink_box(a, k, x, 0, self.X)
        b = __import__("copy").deepcopy(m)
        b.V_[k], b.W_[k] = a.V_[k], a.W_[k]   # the same face moved by hand
        self.assertLess(len(a.members_[k]), len(m.members_[k]))
        cut, gone = refresh_box(b, k, self.X)
        self.assertFalse(gone)
        np.testing.assert_array_equal(a.members_[k], b.members_[k])
        np.testing.assert_allclose(a.sum_[k], b.sum_[k])
        np.testing.assert_allclose(a.centroid_[k], b.centroid_[k])
        self.assertEqual(cut, len(m.members_[k]) - len(b.members_[k]))

    def test_refresh_removes_an_emptied_box(self):
        m = __import__("copy").deepcopy(self.m)
        H = m.n_boxes_
        m.V_[0] = m.W_[0] + 0.01        # no member can be inside now
        cut, gone = refresh_box(m, 0, self.X)
        self.assertTrue(gone)
        self.assertEqual(m.n_boxes_, H - 1)
        self.assertEqual(cut, len(self.m.members_[0]))

    def test_contract_all(self):
        for rule in ("simpson", "gfmm"):
            m, st, log = contract_all(self.m, self.X, rule=rule)
            validate_structure(m, self.X, self.y)
            self.assertIs(type(m), CFMNN)
            self.assertEqual(st["contractions"], len(log))
            self.assertEqual(st["boxes_end"], m.n_boxes_)
            self.assertEqual(st["boxes_start"] - st["boxes_removed"], m.n_boxes_)
            self.assertEqual(st["rows_evicted"], rows_without_box(m, len(self.y)))
            self.assertGreater(st["contractions"], 0)
            # The starting model is untouched.
            validate_structure(self.m, self.X, self.y)
            self.assertEqual(rows_without_box(self.m, len(self.y)), 0)
            # No cross-class pair is left that the rule's own test would contract.
            for j in range(m.n_boxes_):
                for k in range(j + 1, m.n_boxes_):
                    if m.box_class_[j] != m.box_class_[k]:
                        V, W = m.V_.copy(), m.W_.copy()
                        fn = contract_pair if rule == "simpson" else contract_pair_gfmm
                        self.assertIsNone(fn(V, W, j, k), (rule, j, k))

    def test_gfmm_rule_leaves_no_closed_overlap(self):
        m, _, _ = contract_all(self.m, self.X, rule="gfmm")
        V, W, c = m.V_, m.W_, m.box_class_
        for j in range(len(V)):
            o = (np.maximum(V[j], V) <= np.minimum(W[j], W)).all(axis=1) & (c != c[j])
            self.assertFalse(o.any())

    def test_contract_at_errors_order_and_counts(self):
        seed = 7
        t = np.searchsorted(self.m.classes_, self.y)
        errors = np.flatnonzero(self.m.predict_details(self.X).cls != t)
        self.assertGreater(len(errors), 5)
        for budget in (5, None):
            m, st, log = contract_at_errors(self.m, self.X, self.y, budget=budget, seed=seed)
            validate_structure(m, self.X, self.y)
            want = np.random.default_rng(seed).permutation(errors)
            want = want if budget is None else want[:budget]
            self.assertEqual([e["train_index"] for e in log], want.tolist())
            self.assertEqual(st["errors_start"], len(errors))
            self.assertEqual(st["contractions"], sum(e["contracted"] for e in log))
            self.assertEqual(st["errors_tried"], st["contractions"] + st["skipped_already_right"]
                             + st["skipped_no_overlap"] + st["skipped_other"])
            self.assertEqual(st["rows_evicted"], rows_without_box(m, len(self.y)))
        m, st, _ = contract_at_errors(self.m, self.X, self.y, budget=None, seed=seed, rule="gfmm")
        validate_structure(m, self.X, self.y)
        self.assertGreater(st["contractions"], 0)

    def test_rows_in_wrong_class_box(self):
        m = self.m
        t = np.searchsorted(m.classes_, self.y)
        brute = sum(any(m.box_class_[k] != t[r] and (m.V_[k] <= self.X[r]).all()
                        and (self.X[r] <= m.W_[k]).all() for k in range(m.n_boxes_))
                    for r in range(len(self.y)))
        self.assertEqual(rows_in_wrong_class_box(m, self.X, self.y), brute)


class TestScript(unittest.TestCase):
    def fake_split(self, name, seed):
        X, y = toy(seed, n=400)
        return {"train": (X[:300], y[:300]), "test": (X[300:], y[300:])}

    def test_run_one_and_summary(self):
        with mock.patch.object(rc, "split", self.fake_split):
            rows = rc.run_one("thyroid", 3)
        df = pd.DataFrame(rows)
        self.assertEqual(df.arm.tolist(), rc.ARMS)
        b0 = df.set_index("arm").loc["B0"]
        self.assertEqual(b0.test_changed, 0)
        self.assertEqual(b0.rows_evicted, 0)
        self.assertTrue(((df.test_fixed_pct >= 0) & (df.test_broken_pct >= 0)).all())
        self.assertTrue(np.isnan(df.set_index("arm").loc["GFMM", "rows_evicted"]))
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(rc, "split", self.fake_split):
                parts = [pd.DataFrame(rc.run_one("thyroid", s)) for s in (1, 2)]
            runs = pd.concat([df] + parts)
            text = rc.summarise(runs, Path(d))
            self.assertIn("P1 (primary)", text)
            self.assertIn("SR − SC-all", text)
            self.assertTrue((Path(d) / "summary.md").exists())

    def test_verdict(self):
        self.assertEqual(rc.verdict(1.0, 0.01), "higher")
        self.assertEqual(rc.verdict(-1.0, 0.01), "lower")
        self.assertEqual(rc.verdict(-1.0, 0.2), "no significant difference")
        self.assertEqual(rc.verdict(1.0, float("nan")), "no significant difference")

    def test_settings(self):
        self.assertEqual(rc.SEEDS, list(range(120, 130)))
        self.assertEqual(rc.DATASETS, ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"])
        self.assertEqual((rc.THETA, rc.GAMMA, rc.BUDGET, rc.N_ROUNDS, rc.CHOOSER),
                         (0.3, 4.0, 300, 10, "volume"))


if __name__ == "__main__":
    unittest.main()
