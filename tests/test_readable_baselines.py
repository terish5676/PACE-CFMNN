"""Tests for scripts/run_readable_baselines.py: the tie rule, the folds, the size counts and
one small end-to-end run."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_readable_baselines as rb  # noqa: E402


class TestPick(unittest.TestCase):
    def test_best_wins(self):
        self.assertEqual(rb.pick([0.80, 0.90, 0.85]), 1)

    def test_exact_tie_takes_the_simplest(self):
        # Grids are listed simplest first, so the first of the tied settings wins.
        self.assertEqual(rb.pick([0.70, 0.90, 0.90, 0.60]), 1)
        self.assertEqual(rb.pick([0.9, 0.9]), 0)

    def test_float_noise_is_a_tie(self):
        self.assertEqual(rb.pick([0.1 + 0.2, 0.3]), 0)


class TestFolds(unittest.TestCase):
    def test_partition_sorted_and_stratified(self):
        y = np.array([0] * 30 + [1] * 15 + [2] * 6)
        rng = np.random.default_rng(0)
        y = y[rng.permutation(len(y))]
        folds = rb.cv_folds(y, seed=3)
        self.assertEqual(len(folds), 3)
        seen = np.concatenate([va for _, va in folds])
        self.assertEqual(sorted(seen.tolist()), list(range(len(y))))
        for tr, va in folds:
            self.assertTrue(np.all(np.diff(tr) > 0) and np.all(np.diff(va) > 0))   # row order kept
            self.assertEqual(len(np.intersect1d(tr, va)), 0)
            for c, n in [(0, 30), (1, 15), (2, 6)]:
                self.assertEqual(int(np.sum(y[va] == c)), n // 3)

    def test_same_seed_same_folds(self):
        y = np.repeat([0, 1], 20)
        a, b = rb.cv_folds(y, 5), rb.cv_folds(y, 5)
        for (ta, va), (tb, vb) in zip(a, b):
            np.testing.assert_array_equal(va, vb)


class TestSize(unittest.TestCase):
    def test_box_size(self):
        V = np.array([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
        W = np.array([[1.0, 0.5, 0.9], [1.0, 1.0, 1.0]])
        # Box 1 limits features 1 and 2; box 2 covers everything.
        self.assertEqual(rb.box_size(V, W), (2, 1.0))

    def test_tree_size_counts_distinct_features(self):
        X = np.array([[0.1, 0.0], [0.4, 0.0], [0.6, 0.0], [0.9, 0.0],
                      [0.1, 1.0], [0.4, 1.0], [0.6, 1.0], [0.9, 1.0]])
        y = np.array([0, 1, 1, 0, 0, 1, 1, 0])      # needs two cuts on feature 0 only
        t = DecisionTreeClassifier(random_state=0).fit(X, y)
        leaves, cond = rb.tree_size(t)
        self.assertEqual(leaves, t.get_n_leaves())
        self.assertEqual(cond, 1.0)                 # feature 0 tested twice counts once

    def test_tree_single_split(self):
        X = np.array([[0.0], [0.2], [0.8], [1.0]])
        t = DecisionTreeClassifier(random_state=0).fit(X, [0, 0, 1, 1])
        self.assertEqual(rb.tree_size(t), (2, 1.0))


class TestTune(unittest.TestCase):
    def test_tune_returns_a_grid_setting(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(size=(90, 3))
        y = (X[:, 0] > 0.5).astype(int)
        setting, scores = rb.tune("Tree", [4, 8], X, y, seed=0, inner_jobs=1)
        self.assertIn(setting, [4, 8])
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(0 <= s <= 1 for s in scores))


class TestSummary(unittest.TestCase):
    def test_summary_runs_and_reports_verdicts(self):
        import tempfile
        import pandas as pd
        rng = np.random.default_rng(2)
        rows = []
        for ds in ["wine", "digits"]:
            for seed in range(8):
                for m in rb.MODELS:
                    # CFMNN-val is clearly ahead; every other model scores exactly 90.
                    acc = 93.0 + rng.normal(0, 0.1) if m == "CFMNN-val" else 90.0
                    rows.append({"dataset": ds, "seed": seed, "model": m,
                                 "setting": rb.GRIDS[m][0] if m in rb.GRIDS else np.nan,
                                 "accuracy": acc, "macro_f1": acc,
                                 "rules": 5, "conditions_per_rule": 2.0,
                                 "fit_s": 0.1, "predict_s": 0.01})
        with tempfile.TemporaryDirectory() as d:
            text = rb.summarise(pd.DataFrame(rows), Path(d))
            self.assertTrue((Path(d) / "summary.md").exists())
        self.assertIn("**CFMNN-val vs Tree** (accuracy): better 2, level 0, worse 0", text)
        self.assertIn("**CFMNN-0.3 vs Tree** (accuracy): better 0, level 2, worse 0", text)
        self.assertIn("**CFMNN-0.3 vs GFMM-0.3** (accuracy): better 0, level 2, worse 0", text)
        self.assertIn("**CFMNN-0.3 vs GFMM-0.3** (macro_f1)", text)
        self.assertNotIn("**CFMNN-val vs GFMM-0.3**", text)    # tuned vs untuned: not compared


class TestEndToEnd(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("hbbrain"), "hyperbox-brain not installed")
    def test_run_one_on_wine(self):
        rows = rb.run_one("wine", 99, 1, ["CFMNN-0.3", "Tree", "GFMM-0.3"])
        self.assertEqual([r["model"] for r in rows], ["CFMNN-0.3", "Tree", "GFMM-0.3"])
        for r in rows:
            self.assertTrue(0 <= r["accuracy"] <= 100 and 0 <= r["macro_f1"] <= 100)
            self.assertGreater(r["rules"], 0)
        self.assertEqual(rows[0]["setting"], 0.3)
        self.assertIn(rows[1]["setting"], rb.GRIDS["Tree"])
        self.assertEqual(rows[2]["setting"], 0.3)
        self.assertEqual(rows[2]["cv_scores"], "")            # not tuned

    @unittest.skipUnless(importlib.util.find_spec("interpret"), "interpret-core not installed")
    def test_ebm_is_main_effects_only(self):
        rows = rb.run_one("wine", 99, 1, ["EBM"])
        self.assertEqual(rows[0]["rules"], 13)                # one term per feature, no pairs
        self.assertEqual(rows[0]["conditions_per_rule"], 1.0)


if __name__ == "__main__":
    unittest.main()
