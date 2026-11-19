"""Run J: the two sampled baselines, the equal budget and the script."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_sampled_baselines as rj  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.sampled_shap import MarginModel, paired_kernel_shap, permutation_shap  # noqa: E402
from tests.test_refine import problem  # noqa: E402


def small(d=8, seed=1):
    """A small model, 6 points, 40 background rows and the exact Shapley values."""
    X, y = problem(seed, n=500, d=d, classes=3)
    m = CFMNN().fit(X, y)
    Xe, B = X[:6], X[100:140]
    return m, Xe, B, PaceShap(m, B, exact_max_d=d).explain(Xe)


class TestMarginModel(unittest.TestCase):
    def test_equals_margin_score_and_counts_rows(self):
        m, _, _, _ = small()
        Z = np.random.default_rng(0).uniform(size=(300, 8))
        for a in range(len(m.classes_)):
            f = MarginModel(m, a)
            np.testing.assert_allclose(f(Z), xb.margin_score(m, Z, a), atol=1e-12)
            f(Z[0])
            self.assertEqual(f.rows, 301)

    def test_single_class_and_wrong_width(self):
        X = np.random.default_rng(0).uniform(size=(50, 4))
        m = CFMNN().fit(X, np.zeros(50, dtype=int))
        Z = np.random.default_rng(1).uniform(size=(20, 4))
        np.testing.assert_allclose(MarginModel(m, 0)(Z), 2 * m.memberships(Z).max(axis=1) - 1)
        with self.assertRaises(ValueError):
            MarginModel(m, 0)(np.zeros((2, 5)))


class TestPairedKernelShap(unittest.TestCase):
    def test_every_coalition_is_exact(self):
        m, Xe, B, exact = small()
        np.testing.assert_allclose(paired_kernel_shap(m, Xe, B, 2 ** 8), exact, atol=1e-12)

    def test_converges_to_the_exact_values(self):
        m, Xe, B, exact = small()
        err = {k: np.mean([np.abs(paired_kernel_shap(m, Xe, B, k, seed=s) - exact).mean()
                           for s in range(10)]) for k in (30, 240)}
        self.assertLess(err[240], 0.5 * err[30])
        self.assertLess(err[240], 0.05 * np.abs(exact).mean() + 1e-3)

    def test_seeded_efficient_and_counted(self):
        m, Xe, B, _ = small()
        a, used = paired_kernel_shap(m, Xe, B, 60, seed=3, return_rows=True)
        np.testing.assert_array_equal(a, paired_kernel_shap(m, Xe, B, 60, seed=3))
        self.assertFalse(np.array_equal(a, paired_kernel_shap(m, Xe, B, 60, seed=4)))
        np.testing.assert_array_equal(used, 60 * len(B) + len(B) + 1)
        cls = m.predict_details(Xe).cls
        total = [xb.margin_score(m, x[None], c)[0] - xb.margin_score(m, B, c).mean()
                 for x, c in zip(Xe, cls)]
        np.testing.assert_allclose(a.sum(axis=1), total, atol=1e-12)

    def test_same_estimator_as_the_conditional_runs_kernel_shap(self):
        """Same coalitions and solver as ``xai_baselines.kernel_shap_game``; chunks change nothing."""
        m, Xe, B, _ = small()
        ref = xb.kernel_shap_game(m, Xe, B, nsamples=60, seed=5)
        np.testing.assert_allclose(paired_kernel_shap(m, Xe, B, 60, seed=5), ref, atol=1e-12)
        np.testing.assert_allclose(paired_kernel_shap(m, Xe, B, 60, seed=5, chunk_elems=1), ref,
                                   atol=1e-12)

    def test_coalitions_come_in_complementary_pairs(self):
        Z, w = xb.shapley_kernel_coalitions(14, 61, np.random.default_rng(0))
        self.assertEqual(len(Z), 60)
        np.testing.assert_array_equal(Z[1::2], ~Z[0::2])
        np.testing.assert_array_equal(w, 1.0)


class TestPermutationShap(unittest.TestCase):
    def test_converges_to_the_exact_values(self):
        m, Xe, B, exact = small()
        one, many = 17, 64 * 17                       # 1 and 64 orders, each forward and backward
        err = {k: np.mean([np.abs(permutation_shap(m, Xe, B, k, seed=s) - exact).mean()
                           for s in range(4)]) for k in (one, many)}
        self.assertLess(err[many], err[one] / 3)
        self.assertLess(err[many], 0.05 * np.abs(exact).mean() + 1e-3)

    def test_seeded_efficient_counted_and_leaves_the_global_rng_alone(self):
        m, Xe, B, exact = small()
        np.random.seed(123)
        before = np.random.get_state()[1].copy()
        a, used = permutation_shap(m, Xe, B, 170, seed=3, return_rows=True)
        np.testing.assert_array_equal(before, np.random.get_state()[1])
        np.testing.assert_array_equal(a, permutation_shap(m, Xe, B, 170, seed=3))
        self.assertFalse(np.array_equal(a, permutation_shap(m, Xe, B, 170, seed=4)))
        np.testing.assert_allclose(a.sum(axis=1), exact.sum(axis=1), atol=1e-12)
        self.assertLessEqual(used, 170 * len(B))
        self.assertGreater(used, 0.8 * 170 * len(B))


class TestScript(unittest.TestCase):
    def test_setup(self):
        self.assertEqual(rj.SEEDS, list(range(160, 170)))
        self.assertEqual(rj.DATASETS, ["wine", "breast_cancer", "thyroid", "letter", "landsat"])
        self.assertEqual((rj.R_ROWS, rj.P_ORDERS, rj.N_EXPLAIN), (200, 32, 50))
        self.assertEqual(rj.PRIMARY, [("PACE-SHAP", "PermutationSHAP"),
                                      ("PACE-SHAP", "KernelSHAP-paired")])

    def test_budget(self):
        b = rj.budget(200, 16)
        self.assertEqual(b, {"rows": 200 * 32 * 17, "perm_max_evals": 32 * 17,
                             "kernel_coalitions": 32 * 17 - 2})
        # what each baseline can send to the model stays inside the budget
        self.assertLessEqual(b["perm_max_evals"] * 200, b["rows"])
        self.assertLessEqual(b["kernel_coalitions"] * 200 + 200 + 1, b["rows"])
        self.assertEqual(b["perm_max_evals"] // (2 * 16 + 1), 16)      # 16 pairs = 32 orders

    def toy(self):
        X, y = problem(2, n=500, d=13, classes=3)
        m = CFMNN().fit(X[:400], y[:400])
        return m, X[400:408], X[:60], X[60:120]          # model, points, evaluation rows, B

    def test_every_method_keeps_to_the_budget(self):
        m, Xe, _, B = self.toy()
        b = rj.budget(len(B), 13)
        for name, fn in rj.explainers(m, Xe, B).items():
            attr, used = fn(0)
            self.assertEqual(attr.shape, Xe.shape)
            self.assertLessEqual(used, b["rows"], name)
            if name in (rj.PERM, rj.KERNEL):
                self.assertGreater(used, 0.8 * b["rows"], name)

    def test_run_one_rows(self):
        with mock.patch.object(rj, "prepare", return_value=self.toy()), \
                mock.patch.object(rj, "N_EXACT", 3):
            rows = rj.run_one("toy", 0)
        self.assertEqual([r["method"] for r in rows], rj.METHODS)
        self.assertTrue(all(list(r) == list(rows[0]) for r in rows))
        by = {r["method"]: r for r in rows}
        self.assertEqual(by[rj.PACE]["n_exact"], 3)
        for name in (rj.PACE, rj.PERM, rj.KERNEL):
            self.assertLess(by[name]["rel_l1_error"], 0.5 * by[rj.RANDOM]["rel_l1_error"])
            self.assertLess(by[name]["margin_auc"], by[rj.RANDOM]["margin_auc"])
            self.assertGreater(by[name]["stability_top3"], by[rj.RANDOM]["stability_top3"])

    def test_pace_shap_is_v3s_call(self):
        """Run J's PACE-SHAP is the call of the second explanation run (``methods_v3``)."""
        m, Xe, _, B = self.toy()
        got, _ = rj.explainers(m, Xe, B)[rj.PACE](7)
        want = rj.rf.methods_v3(m, Xe, None, B)["PACE-SHAP"](7)
        np.testing.assert_array_equal(got, want)

    def test_compare_reads_lower_as_better(self):
        rng = np.random.default_rng(0)
        rows = [{"dataset": ds, "seed": s, "method": meth,
                 "margin_auc": base + rng.normal(0, 0.001)}
                for ds, gap in (("wine", -0.05), ("letter", 0.05), ("landsat", 0.0))
                for s in range(160, 170)
                for meth, base in ((rj.PACE, 0.2 + gap), (rj.PERM, 0.2))]
        df = pd.DataFrame(rows)
        low = rj.compare(df, rj.PACE, rj.PERM, "margin_auc", lower=True).set_index("dataset")
        self.assertEqual(list(low.verdict[["wine", "letter", "landsat"]]), ["better", "worse", "level"])
        self.assertEqual(low.loc["wine", "seeds_in_favour"], 10)
        high = rj.compare(df, rj.PACE, rj.PERM, "margin_auc", lower=False).set_index("dataset")
        self.assertEqual(list(high.verdict[["wine", "letter"]]), ["worse", "better"])


if __name__ == "__main__":
    unittest.main()
