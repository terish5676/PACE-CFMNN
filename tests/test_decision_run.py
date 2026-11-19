"""Tests for scripts/run_decision.py (Run P5): the comparison's direction and Holm step, the check,
and one small end-to-end task."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_decision as rd  # noqa: E402


def _fake(gap=-0.1, ds=("magic", "page_blocks")):
    rows = []
    for k, name in enumerate(ds):
        for s in range(10):
            for p in range(4):
                for m, v in ((rd.DEC, 0.5 + (gap if k == 0 else 0.0) + 0.001 * s), (rd.SHP, 0.5 + 0.001 * s)):
                    rows.append({"dataset": name, "seed": s, "point": p, "method": m, "tied": p < 2,
                                 "decision_deletion_auc": v, "margin_deletion_auc": 0.3,
                                 "decision_insertion_auc": 0.7, "margin_insertion_auc": 0.6,
                                 "check_max_diff": 0.0})
    return pd.DataFrame(rows)


class TestCompare(unittest.TestCase):
    def test_direction_and_holm(self):
        t = rd.compare(_fake(-0.1), "decision_deletion_auc", True).set_index("dataset")
        self.assertEqual(t.loc["magic", "verdict"], "better")
        self.assertEqual((t.loc["magic", "wins"], t.loc["magic", "losses"]), (10, 0))
        self.assertAlmostEqual(t.loc["magic", "p_holm"], 2 * 0.001953125)      # Holm over 2 datasets
        self.assertEqual(t.loc["page_blocks", "verdict"], "no significant difference")
        t = rd.compare(_fake(+0.1), "decision_deletion_auc", True).set_index("dataset")
        self.assertEqual(t.loc["magic", "verdict"], "worse")
        t = rd.compare(_fake(+0.1), "decision_deletion_auc", False).set_index("dataset")
        self.assertEqual(t.loc["magic", "verdict"], "better")

    def test_check_and_summary(self):
        df = _fake()
        self.assertTrue(rd.check(df)[0])
        bad = df.copy()
        bad.loc[0, "check_max_diff"] = 1e-6
        self.assertFalse(rd.check(bad)[0])
        with tempfile.TemporaryDirectory() as tmp:
            text = rd.summarise(df, Path(tmp))
        self.assertIn("## Primary: decision deletion AUC, tied points (lower is better)", text)


class TestEndToEnd(unittest.TestCase):
    def test_one_small_task(self):
        df = pd.DataFrame(rd.run_one("page_blocks", 194, 3))
        self.assertEqual(sorted(df.method.unique()), sorted([rd.DEC, rd.SHP]))
        self.assertTrue(rd.check(df)[0])
        self.assertTrue(np.isfinite(df.check_max_diff).all())
        self.assertTrue(df.decision_deletion_auc.between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
