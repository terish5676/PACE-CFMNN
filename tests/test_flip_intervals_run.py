"""Tests for scripts/run_flip_intervals.py (Run P7): the dense check finds nothing wrong on a toy
model, catches a wrong flip set, and the summary counts what the 32 values miss."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_flip_intervals as fi  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.flip_exact import FlipSet, flip_sets  # noqa: E402


def _toy(seed=3, d=4):
    rng = np.random.default_rng(seed)
    X = rng.random((300, d))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.8).astype(int) + (X[:, 2] > 0.7)
    return CFMNN(theta=0.3).fit(X, y), X


class TestDenseCheck(unittest.TestCase):
    def test_exact_sets_pass_on_a_toy_model(self):
        m, X = _toy()
        flipping = 0
        for x in X[:15]:
            for fs in flip_sets(m, x):
                outside, missed, nf, ni = fi.check_pair(m, x, fs, grid=2001)
                self.assertEqual((outside, missed), (0, 0))
                flipping += ni > 0
        self.assertGreater(flipping, 0)

    def test_breaks_stay_in_the_domain(self):
        # regression: box-edge breakpoints below 0 or above 1 once made intervals outside the domain
        m, X = _toy()
        for x in X[:15]:
            for fs in flip_sets(m, x):
                lo, hi = min(0.0, x[fs.feature]), max(1.0, x[fs.feature])
                self.assertGreaterEqual(fs.breaks.min(), lo)
                self.assertLessEqual(fs.breaks.max(), hi)
                for a, b in fs.flip_intervals():
                    self.assertTrue(lo <= a <= b <= hi)

    def test_a_wrong_set_is_caught(self):
        m, X = _toy()
        for x in X[:30]:
            for fs in flip_sets(m, x):
                if not fs.flip_intervals():
                    continue
                # claim the class never changes: every grid flip is now outside the set
                empty = FlipSet(fs.feature, fs.breaks, np.full_like(fs.point_cls, fs.cls_x),
                                np.full_like(fs.open_cls, fs.cls_x), fs.cls_x)
                outside, _, nf, _ = fi.check_pair(m, x, empty, grid=2001)
                if nf:
                    self.assertGreater(outside, 0)
                    return
        self.fail("no flipping pair on the toy model")

    def test_in_set_matches_the_model(self):
        m, X = _toy()
        rng = np.random.default_rng(0)
        for x in X[:10]:
            for fs in flip_sets(m, x):
                u = np.r_[rng.random(50), fs.breaks]
                P = np.repeat(x[None], len(u), axis=0)
                P[:, fs.feature] = u
                truth = m.predict_details(P).cls != fs.cls_x
                np.testing.assert_array_equal(fi._in_set(fs, u), truth)


def _fake(missed32=1, outside=0):
    rows = []
    for p in range(4):
        for i in range(2):
            flips = (p + i) % 2 == 0
            rows.append({"dataset": "S1", "seed": 1, "point": p, "feature": i, "d": 2, "boxes": 50,
                         "n_intervals": int(flips), "exact_measure": 0.1 * flips, "exact_flips": flips,
                         "grid32_flips": flips and not (p == 0 and i == 0 and missed32),
                         "grid32_share": 0.1, "grid32_agrees": True, "grid32_at_break": 0, "dist_exact": 0.1 if flips else np.inf,
                         "dist_grid32": 0.2 if flips else np.inf,
                         "point_dist_exact": np.nan, "point_dist_grid32": np.nan, "dense": p < 2,
                         "dense_outside": outside if p < 2 else np.nan,
                         "dense_missed": 0 if p < 2 else np.nan, "dense_flips": 5 if p < 2 else np.nan})
        rows[-1]["point_dist_exact"], rows[-1]["point_dist_grid32"] = 0.1, 0.2
    return pd.DataFrame(rows)


class TestCheckAndSummary(unittest.TestCase):
    def test_check(self):
        self.assertTrue(fi.check(_fake())[0])
        self.assertFalse(fi.check(_fake(outside=2))[0])
        df = _fake()
        df.loc[3, "grid32_agrees"] = False
        self.assertFalse(fi.check(df)[0])

    def test_summary_counts_the_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = fi.summarise(_fake(), Path(tmp))
        self.assertIn("**Check: PASS.**", text)
        row = [line for line in text.splitlines() if line.startswith("| S1 |")][0]
        cells = [c.strip() for c in row.strip("|").split("|")]
        self.assertEqual(cells[3:7], ["8", "4", "1", "25.0%"])


class TestEndToEnd(unittest.TestCase):
    def test_one_small_task_on_a_smoke_seed(self):
        old = fi.N_DENSE
        fi.N_DENSE = 1
        try:
            rows = fi.run_one("S1", 194, 2)
        finally:
            fi.N_DENSE = old
        df = pd.DataFrame(rows)
        self.assertTrue(fi.check(df)[0])
        self.assertEqual(int(df.dense.sum()), df.d.iloc[0])


if __name__ == "__main__":
    unittest.main()
