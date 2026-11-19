"""Tests for scripts/run_skin_dedup.py (Run P3): the reference arm reproduces the refinement run's
B0 and SR, the reading follows the pre-registered rule, and the summary reports the primary test."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_skin_dedup as sd  # noqa: E402


class TestReference(unittest.TestCase):
    def test_reproduces_the_refinement_run(self):
        ref = pd.read_csv(ROOT / "results" / "refinement_v2_confirmatory" / "runs.csv")
        ref = ref[(ref.dataset == "thyroid") & (ref.seed == 50)].set_index("arm").test_macro_f1
        rows = pd.DataFrame(sd.part_reference("thyroid", 50)).set_index("arm").test_macro_f1
        for arm in ("B0", "SR"):
            self.assertAlmostEqual(rows[arm], ref[arm], places=9)

    def test_datasets(self):
        self.assertEqual(sd.dataset_for("reference"), "skin")
        self.assertEqual({sd.dataset_for(p) for p in sd.PARTS if p != "reference"}, {"skin_dedup"})


class TestReading(unittest.TestCase):
    def test_four_outcomes(self):
        self.assertIn("stands", sd.reading({"p": 0.01, "diff": 3.0}))
        self.assertIn("inflated", sd.reading({"p": 0.01, "diff": 1.0}))
        self.assertIn("hurts", sd.reading({"p": 0.01, "diff": -1.0}))
        self.assertIn("No significant difference", sd.reading({"p": 0.2, "diff": 2.0}))
        self.assertAlmostEqual(sd.THRESHOLD, 2.83)

    def test_paired_test(self):
        piv = pd.DataFrame({"SR": np.arange(10) + 1.0, "B0": np.zeros(10)})
        t = sd.paired_test(piv, "SR", "B0")
        self.assertEqual((t["wins"], t["losses"], t["n"]), (10, 0, 10))
        self.assertAlmostEqual(t["p"], 0.001953125)
        self.assertEqual(sd.paired_test(pd.DataFrame({"SR": np.ones(5), "B0": np.ones(5)}), "SR", "B0")["p"], 1.0)


class TestSummary(unittest.TestCase):
    def test_primary_line(self):
        rows = []
        for s in sd.SEEDS:
            for arm, f1 in (("B0", 90.0), ("SR", 91.0 + 0.1 * (s - 260)), ("SR+AL", 91.5)):
                rows.append({"dataset": sd.NAME, "seed": s, "arm": arm, "test_macro_f1": f1, "test_accuracy": f1,
                             "test_min_class_recall": f1, "total_labels": 0, "churn_broken_share": 0.1})
        with tempfile.TemporaryDirectory() as tmp:
            text = sd.summarise({"refinement": pd.DataFrame(rows)}, Path(tmp))
        self.assertIn("**+1.45 points** (SR better on 10, worse on 0)", text)
        self.assertIn("inflated", text)
        self.assertIn("51,444 of 245,057 rows", text)


if __name__ == "__main__":
    unittest.main()
