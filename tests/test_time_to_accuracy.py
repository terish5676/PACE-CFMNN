"""Tests for scripts/run_time_to_accuracy.py (Run P1): KernelSHAP with every coalition is exact,
b* and the bootstrap ratio are computed as described, and the summary reads them correctly."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_time_to_accuracy as ta  # noqa: E402


class TestPieces(unittest.TestCase):
    def test_rel_l1(self):
        self.assertAlmostEqual(ta.rel_l1(np.array([1.0, -1.0]), np.array([2.0, -2.0])), 0.5)
        self.assertEqual(ta.rel_l1(np.zeros(3), np.zeros(3)), 0.0)

    def test_b_star_is_the_smallest_budget_under_the_target(self):
        rows = []
        for p in range(5):
            rows.append({"arm": ta.PACE, "budget": "exact", "rel_l1": 0.0})
            for b, e in (("100", 0.05), ("200", 0.02), ("400", 0.008), ("700", 0.004), ("all", 0.0)):
                rows.append({"arm": ta.KERNEL, "budget": b, "rel_l1": e + 0.001 * p})
        bs, med = ta.b_star(pd.DataFrame(rows))
        self.assertEqual(bs, "400")
        self.assertAlmostEqual(med["400"], 0.010)
        rows = [r for r in rows if r["budget"] not in ("400", "700")]
        self.assertEqual(ta.b_star(pd.DataFrame(rows))[0], "all")

    def test_ratio_ci(self):
        r, lo, hi = ta.ratio_ci([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(r, 10.0)
        self.assertAlmostEqual(lo, 10.0)
        self.assertAlmostEqual(hi, 10.0)
        r, lo, hi = ta.ratio_ci(np.r_[np.full(10, 10.0), np.full(10, 30.0)], np.full(20, 2.0))
        self.assertAlmostEqual(r, 10.0)
        self.assertLess(lo, r)
        self.assertGreater(hi, r)


def _fake():
    rows = []
    for s in (300, 301):
        for p in range(3):
            base = {"part": "a", "dataset": "S1", "seed": s, "d": 10, "boxes": 100, "background_rows": 200,
                    "point": p}
            rows.append({**base, "arm": ta.PACE, "budget": "exact", "ms": 10.0, "rel_l1": 0.0})
            for b, e, ms in (("100", 0.05, 20.0), ("200", 0.03, 40.0), ("400", 0.009, 80.0),
                             ("700", 0.005, 140.0), ("all", 0.0, 200.0)):
                rows.append({**base, "arm": ta.KERNEL, "budget": b, "ms": ms, "rel_l1": e})
    for ds, d, k in (("S1", 10, 50.0), ("wine", 13, 30.0)):
        for p in range(3):
            base = {"part": "b", "dataset": ds, "seed": 303, "d": d, "boxes": 100, "background_rows": 200,
                    "point": p, "rel_l1": np.nan}
            rows.append({**base, "arm": ta.PACE, "budget": "default", "ms": 10.0})
            rows.append({**base, "arm": ta.KERNEL, "budget": "200", "ms": k})
    return pd.DataFrame(rows)


class TestSummary(unittest.TestCase):
    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = ta.summarise(_fake(), Path(tmp))
        row = [line for line in text.splitlines() if line.startswith("| S1 | 10 | 5.00%")][0]
        cells = [c.strip() for c in row.strip("|").split("|")]
        self.assertEqual(cells[7:12], ["400", "10", "80", "8.0× (8.0–8.0)", "4.0×"])
        self.assertIn("KernelSHAP took 3.0–5.0× as long", text)
        self.assertIn("| wine | 13 | 32 permutations |", text)


class TestEndToEnd(unittest.TestCase):
    def test_every_coalition_is_exact(self):
        rows = pd.DataFrame(ta.run_one("S1", 194, 1, "a"))
        self.assertEqual(sorted(rows.budget.unique()), ["100", "200", "400", "700", "all", "exact"])
        self.assertLess(rows[rows.budget == "all"].rel_l1.max(), 1e-9)
        self.assertGreater(rows[rows.budget == "100"].rel_l1.max(), 0.0)
        self.assertTrue((rows.ms > 0).all())

    def test_part_b(self):
        rows = pd.DataFrame(ta.run_one("wine", 194, 1, "b"))
        self.assertEqual(sorted(rows.arm), sorted([ta.PACE, ta.KERNEL]))


if __name__ == "__main__":
    unittest.main()
