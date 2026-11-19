"""Tests for scripts/run_recurrence.py (Run P8): the four arms give the same values, the check
passes and fails as it should, and the summary's ratios and decision line are the ones described."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_recurrence as rr  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import PaceShap, brute_force_shap  # noqa: E402


def _toy(d=5):
    rng = np.random.default_rng(11)
    X = rng.random((300, d))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.8).astype(int) + (X[:, 2] > 0.7)
    m = CFMNN(theta=0.3).fit(X, y)
    return m, X[:30], rng.random((4, d))


class TestSameValues(unittest.TestCase):
    def test_four_ways_agree(self):
        m, B, Xe = _toy()
        a = PaceShap(m, B).explain(Xe)
        b = PaceShap(m, B, exact_algo="recurrence").explain(Xe)
        self.assertLessEqual(np.abs(a - b).max(), rr.TOL_SAME)
        self.assertLessEqual(np.abs(a - brute_force_shap(m, Xe, B)).max(), rr.TOL_SAME)
        self.assertLessEqual(np.abs(b - xb.exact_shap(m, Xe, B)).max(), rr.TOL_EXACT)

    def test_above_twelve_features_with_a_raised_limit(self):
        m, B, Xe = _toy(d=13)
        b = PaceShap(m, B[:8], exact_max_d=13, exact_algo="recurrence").explain(Xe[:1])
        self.assertLessEqual(np.abs(b - brute_force_shap(m, Xe[:1], B[:8])).max(), rr.TOL_SAME)


def _fake(diff_recur=0.0, diff_brute=0.0, diff_exact=1e-15, letter_recur_ms=3000.0):
    rows = []
    for ds, d in (("S1", 10), ("letter", 16)):
        for s in (1, 2):
            arms = [(rr.MATMUL, 80.0, 0.0), (rr.RECUR, 20.0, diff_recur), (rr.BRUTE, 400.0, diff_brute)]
            if ds == "letter":
                arms = [(rr.MATMUL, 4 * letter_recur_ms, 0.0), (rr.RECUR, letter_recur_ms, diff_recur),
                        (rr.BRUTE, 100 * letter_recur_ms, diff_brute), (rr.EXACT, 200 * letter_recur_ms, 0.0)]
            for arm, ms, dif in arms:
                rows.append({"dataset": ds, "seed": s, "arm": arm, "d": d, "boxes": 1500,
                             "background_rows": 200, "n": 10, "ms_per_explanation": ms,
                             "max_abs_diff_matmul": dif,
                             "max_abs_diff_exact": diff_exact if ds == "letter" else np.nan,
                             "max_abs_value": 0.3})
    return pd.DataFrame(rows)


class TestCheckAndSummary(unittest.TestCase):
    def test_check_passes_within_the_limits(self):
        ok, bad = rr.check(_fake())
        self.assertTrue(ok)
        self.assertEqual(bad, [])

    def test_check_fails_outside_the_limits(self):
        self.assertFalse(rr.check(_fake(diff_recur=1e-11))[0])
        self.assertFalse(rr.check(_fake(diff_brute=1e-11))[0])
        self.assertFalse(rr.check(_fake(diff_exact=1e-8))[0])

    def test_summary_ratios(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = rr.summarise(_fake(), Path(tmp))
        self.assertIn("**Check: PASS.**", text)
        row = [line for line in text.splitlines() if line.startswith("| S1 |")][0]
        cells = [c.strip() for c in row.strip("|").split("|")]
        self.assertEqual(cells[3:9], ["80.0", "20.0", "400.0", "–", "4.00×", "20.0×"])

    def test_decision_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("**yes**", rr.summarise(_fake(letter_recur_ms=3000.0), Path(tmp)))
            self.assertIn("**no**", rr.summarise(_fake(letter_recur_ms=9000.0), Path(tmp)))


class TestEndToEnd(unittest.TestCase):
    def test_one_small_task_on_a_smoke_seed(self):
        rows = rr.run_one("shuttle", 194, 1)
        df = pd.DataFrame(rows)
        self.assertEqual(sorted(df.arm), sorted([rr.MATMUL, rr.RECUR, rr.BRUTE]))
        self.assertTrue(rr.check(df)[0])
        self.assertTrue((df.ms_per_explanation > 0).all())

    def test_exact_comparison_uses_only_the_points_both_explained(self):
        # Amendment 1: when ExactExplainer explains fewer points than PACE-SHAP (Letter), the
        # comparison must use only the shared points, not broadcast one point against all of them.
        few, large = rr.FEW, rr.LARGE
        try:
            rr.FEW, rr.LARGE = {"shuttle": 1}, ["shuttle"]
            df = pd.DataFrame(rr.run_one("shuttle", rr.SEEDS[0], 3))
        finally:
            rr.FEW, rr.LARGE = few, large
        pace = df[df.arm.isin([rr.MATMUL, rr.RECUR])]
        self.assertEqual(sorted(pace.n), [3, 3])
        self.assertEqual(int(df[df.arm == rr.EXACT].n.iloc[0]), 1)
        self.assertTrue((pace.max_abs_diff_exact <= rr.TOL_EXACT).all())
        self.assertTrue(rr.check(df)[0])


if __name__ == "__main__":
    unittest.main()
