"""Run C: v3's methods unchanged plus the like-for-like KernelSHAP arm, the tie-aware columns and
the summary."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_faithfulness as rf  # noqa: E402
import run_more_points as mp  # noqa: E402

N = 6


class TestMethods(unittest.TestCase):
    def test_v3_methods_plus_kernelshap_on_the_same_rows(self):
        _, Xtr, Xte, model, _ = rf.setup("wine", 120, N)
        _, bex = rf.backgrounds_v3(Xtr, 120)
        fns = mp.methods_c(model, Xte, Xtr, bex, 120)
        v3 = rf.methods_v3(model, Xte, Xtr, bex)
        self.assertEqual(list(fns), mp.METHODS)
        self.assertEqual(list(fns)[:-1], list(v3))               # v3's methods, in v3's order
        self.assertEqual(list(fns)[-1], "KernelSHAP-same")
        self.assertEqual(sorted(mp.METHODS), sorted(list(v3) + ["KernelSHAP-same"]))
        # the added arm uses all of PACE-SHAP's explainer rows, not v3's 20
        same = rf._same_rows(bex, rf.SHAP_V3_R, 120)
        self.assertTrue(np.array_equal(same, bex[:rf.SHAP_V3_R]))
        self.assertGreater(len(same), rf.SHAP_BG_ROWS)


class TestRunOne(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = mp.run_one("wine", 120, n=N)
        cls.df = pd.DataFrame(cls.rows)

    def test_columns_methods_and_ties(self):
        df = self.df
        self.assertEqual(list(df.method), mp.METHODS)
        self.assertEqual(set(df.protocol), {"v3-200"})
        self.assertEqual(set(df.n), {N})
        self.assertEqual(df.n_tied.nunique(), 1)               # the same points for every method
        self.assertEqual(df.n_near_tie.nunique(), 1)
        self.assertTrue((df.n_near_tie >= df.n_tied).all())
        for col in ("margin_auc_raw", "insertion_margin_auc_raw", "margin_auc_untied",
                    "insertion_margin_auc_untied", "deletion_auc", "insertion_auc"):
            self.assertIn(col, df.columns)
        self.assertTrue(df.loc[df.method == "KernelSHAP-strong", "stability_top3"].isna().all())
        self.assertTrue(df.loc[df.method != "KernelSHAP-strong", "stability_top3"].between(0, 1).all())

    def test_the_v3_methods_reproduce_v3_exactly(self):
        v3, _ = rf.run("wine", 120, N, "v3")
        v3 = {r["method"]: r for r in v3}
        mine = {r["method"]: r for r in self.rows}
        for m, ref in v3.items():
            for col in ("margin_auc", "insertion_margin_auc", "deletion_auc", "insertion_auc",
                        "margin_auc_abs"):
                self.assertAlmostEqual(mine[m][col], ref[col], places=12, msg=f"{m} {col}")


class TestSummary(unittest.TestCase):
    def table(self, shift):
        recs = []
        for ds in ("wine", "thyroid"):
            for s in range(110, 120):
                for m in mp.METHODS:
                    base = 0.5 + 0.001 * s
                    rec = {c: 0.0 for c in ("n", "n_tied", "n_near_tie", "deletion_auc", "margin_auc",
                                            "margin_auc_abs", "insertion_auc", "insertion_margin_auc",
                                            "margin_auc_raw", "insertion_margin_auc_raw",
                                            "margin_auc_untied", "insertion_margin_auc_untied",
                                            "margin_auc_raw_untied", "insertion_margin_auc_raw_untied",
                                            "share_negative", "stability_top3", "ms_per_explanation")}
                    rec.update(dataset=ds, seed=s, method=m, protocol="v3-200", n=50, n_tied=5,
                               n_near_tie=6, gt_top3_precision=np.nan)
                    for col in ("margin_auc", "margin_auc_untied", "margin_auc_raw"):
                        rec[col] = base + (shift if m == "PACE-SHAP" else 0.0) + 0.0001 * (s % 3)
                    for col in ("insertion_margin_auc", "insertion_margin_auc_untied",
                                "insertion_margin_auc_raw"):
                        rec[col] = base - (shift if m == "PACE-SHAP" else 0.0) + 0.0001 * (s % 3)
                    recs.append(rec)
        return pd.DataFrame(recs)

    def test_primary_verdicts_and_tie_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = mp.summarise(self.table(-0.05), Path(tmp))     # PACE-SHAP lower deletion, higher insertion
            self.assertTrue((Path(tmp) / "summary.md").exists())
        self.assertIn("PACE-SHAP minus KernelSHAP, margin_auc** (negative favours PACE-SHAP): "
                      "PACE-SHAP is better on 2, level on 0, worse on 0 of 2 datasets.", text)
        self.assertIn("PACE-SHAP minus KernelSHAP-same, insertion_margin_auc** (positive favours PACE-SHAP): "
                      "PACE-SHAP is better on 2", text)
        self.assertIn("Tied and near-tied points", text)
        self.assertIn("untied points only", text)
        with tempfile.TemporaryDirectory() as tmp:
            text = mp.summarise(self.table(0.05), Path(tmp))      # the other way round
        self.assertIn("PACE-SHAP is better on 0, level on 0, worse on 2 of 2 datasets.", text)
        with tempfile.TemporaryDirectory() as tmp:
            text = mp.summarise(self.table(0.0), Path(tmp))
        self.assertIn("PACE-SHAP is better on 0, level on 2, worse on 0 of 2 datasets.", text)


if __name__ == "__main__":
    unittest.main()
