"""Run L: Covertype with theta chosen by validation, then self-repair (made-up data, no download)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import run_covertype as rk  # noqa: E402
import run_covertype_theta as rl  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from run_readable_baselines import GRIDS  # noqa: E402
from test_covertype import fake_covertype  # noqa: E402


class TestSetup(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(rl.SEEDS, [210, 211, 212, 213, 214])
        self.assertEqual(rl.MODELS, ["B0-0.3", "SR-0.3", "B0-val", "SR-val"])
        self.assertEqual((rl.GAMMA, rl.FIXED_THETA), (4.0, 0.3))
        self.assertEqual((rl.BUDGET, rl.N_ROUNDS, rl.CHOOSER), (300, 10, "volume"))
        self.assertEqual(GRIDS["CFMNN-val"], [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05])
        self.assertIs(rl.prepare, rk.prepare)                     # Run K's split, unchanged
        self.assertEqual(rl.SUBSAMPLE, 0.1)                       # the study's share of the rows
        self.assertEqual((rl.TUNE_ROWS, rl.GATE_ROWS), (None, None))   # all rows of that share

    def test_stratified_rows(self):
        y = np.repeat([1, 2, 3], [600, 300, 100])
        self.assertTrue(np.array_equal(rl.stratified_rows(y, None, 0), np.arange(1000)))
        self.assertTrue(np.array_equal(rl.stratified_rows(y, 5000, 0), np.arange(1000)))
        idx = rl.stratified_rows(y, 200, 1)
        self.assertEqual(len(idx), 200)
        self.assertTrue(np.all(np.diff(idx) > 0))                 # sorted, no repeats
        self.assertEqual(np.bincount(y[idx])[1:].tolist(), [120, 60, 20])
        self.assertTrue(np.array_equal(idx, rl.stratified_rows(y, 200, 1)))   # seeded


class TestTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = fake_covertype(1200, seed=3)
        cls.rows = rl.run_one("covertype", 5, ds=cls.ds, tune_rows=600, gate_rows=500)
        cls.df = pd.DataFrame(cls.rows)

    def test_four_rows_same_columns(self):
        self.assertEqual(list(self.df.columns), rl.COLUMNS)
        self.assertEqual(self.df.model.tolist(), rl.MODELS)
        self.assertTrue((self.df.tune_rows == 600).all() and (self.df.gate_rows == 500).all())

    def test_b0_is_a_cfmnn_fitted_directly(self):
        Xtr, ytr, Xte, yte, _ = rk.prepare(5, ds=self.ds)
        for model, theta in (("B0-0.3", 0.3), ("B0-val", float(self.df.theta.iloc[2]))):
            m = CFMNN(theta=theta, gamma=4.0).fit(Xtr, ytr)
            r = self.df[self.df.model == model].iloc[0]
            self.assertAlmostEqual(r.test_accuracy, 100 * float(np.mean(m.predict(Xte) == yte)), places=9)
            self.assertEqual(int(r.boxes), m.n_boxes_)

    def test_theta_from_the_grid_and_repair_recorded(self):
        self.assertIn(float(self.df.theta.iloc[2]), GRIDS["CFMNN-val"])
        self.assertEqual(len(self.df.cv_scores.iloc[0].split(";")), len(GRIDS["CFMNN-val"]))
        sr = self.df[self.df.model.str.startswith("SR")]
        self.assertTrue((sr.edits_logged <= rl.BUDGET).all() and (sr.edits_accepted <= sr.edits_logged).all())
        self.assertTrue(sr.fit_seconds.isna().all() and sr.repair_seconds.notna().all())
        b0 = self.df[self.df.model.str.startswith("B0")]
        self.assertTrue(b0.edits_logged.isna().all() and b0.fit_seconds.notna().all())
        self.assertTrue((self.df.pace_shap_mode == "sampled").all())   # 54 features
        share = self.df.train_tie_share_pct[b0.index]          # Proposition 1 before repair
        self.assertTrue(((share == 100) | (b0.train_errors == 0)).all())     # no errors -> no share
        self.assertTrue(share[b0.train_errors == 0].isna().all())

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = rl.summarise(self.df, Path(tmp))
            self.assertTrue((Path(tmp) / "summary.md").exists())
        self.assertIn("Descriptive: no comparison is tested", text)
        for m in rl.MODELS:
            self.assertIn(f"| {m} |", text)


if __name__ == "__main__":
    unittest.main()
