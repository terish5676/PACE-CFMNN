"""Tests for scripts/run_speed_source.py (Run P): the three arms give the same values, the check
passes and fails as it should, and the summary's ratios are the ones described."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_speed_source as ss  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402


def _toy():
    rng = np.random.default_rng(7)
    X = rng.random((300, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.8).astype(int) + (X[:, 2] > 0.7)
    m = CFMNN(theta=0.3).fit(X, y)
    return m, X[:40], rng.random((6, 5))


class TestSameValues(unittest.TestCase):
    def test_pruning_changes_nothing_and_matches_shap(self):
        m, B, Xe = _toy()
        on = PaceShap(m, B).explain(Xe)
        off = PaceShap(m, B, prune=False).explain(Xe)
        self.assertLessEqual(np.abs(on - off).max(), ss.TOL_NOPRUNE)
        self.assertLessEqual(np.abs(on - xb.exact_shap(m, Xe, B)).max(), ss.TOL_EXACT)

    def test_pruning_skips_some_pairs(self):
        m, B, Xe = _toy()
        ps = PaceShap(m, B)
        ps.explain(Xe)
        self.assertGreater(ps.stats_["share_of_boxes_skipped"], 0.0)
        off = PaceShap(m, B, prune=False)
        off.explain(Xe)
        self.assertEqual(off.stats_["share_of_boxes_skipped"], 0.0)


def _fake(diff_noprune=0.0, diff_exact=1e-16):
    rows = []
    for s in (1, 2):
        for arm, ms, dif in ((ss.EXACT_ARM, 1000.0, diff_exact), (ss.NOPRUNE_ARM, 100.0, diff_noprune),
                             (ss.PACE_ARM, 50.0, 0.0)):
            rows.append({"dataset": "S1", "seed": s, "arm": arm, "d": 10, "boxes": 1500,
                         "background_rows": 200, "n": 2, "ms_per_explanation": ms, "max_abs_diff": dif,
                         "max_abs_value": 0.3,
                         "share_pairs_skipped": 0.6 if arm == ss.PACE_ARM else np.nan})
    return pd.DataFrame(rows)


class TestCheckAndSummary(unittest.TestCase):
    def test_check_passes_within_the_limits(self):
        ok, bad = ss.check(_fake())
        self.assertTrue(ok)
        self.assertEqual(bad, [])

    def test_check_fails_outside_the_limits(self):
        self.assertFalse(ss.check(_fake(diff_noprune=1e-10))[0])
        self.assertFalse(ss.check(_fake(diff_exact=1e-8))[0])

    def test_summary_ratios(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = ss.summarise(_fake(), Path(tmp))
        self.assertIn("**Check: PASS.**", text)
        row = [line for line in text.splitlines() if line.startswith("| S1 |")][0]
        cells = [c.strip() for c in row.strip("|").split("|")]
        self.assertEqual(cells[6:10], ["10.0×", "2.0×", "20.0×", "0.600"])


class TestEndToEnd(unittest.TestCase):
    def test_one_small_task_on_a_smoke_seed(self):
        rows = ss.run_one("shuttle", 194, 1)
        df = pd.DataFrame(rows)
        self.assertEqual(sorted(df.arm), sorted(ss.ARMS))
        self.assertTrue(ss.check(df)[0])
        self.assertTrue((df.ms_per_explanation > 0).all())


if __name__ == "__main__":
    unittest.main()
