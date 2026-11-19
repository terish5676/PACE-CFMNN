"""Run D: the effective-rows measure, the settings, one small end-to-end run and the summary."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_conditional_settings as cs  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.explain import PaceShap, cond_scale  # noqa: E402


class TestEffectiveRows(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.B = rng.uniform(size=(200, 5))
        self.X = rng.uniform(size=(6, 5))
        self.scale = cond_scale(self.B)

    def test_limits_and_monotone_in_sigma(self):
        eff = {s: cs.effective_rows_sampled(self.X, self.B, self.scale, s, np.random.default_rng(1))
               for s in (0.01, 0.05, 0.2, 0.4, 100.0)}
        self.assertLess(eff[0.01][0], 3.0)                           # one row carries the weight
        self.assertAlmostEqual(eff[100.0][0], 200.0, delta=0.5)      # every row counts
        self.assertAlmostEqual(eff[100.0][1], 200.0, delta=0.5)
        for a, b in ((0.01, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 100.0)):
            self.assertLessEqual(eff[a][0], eff[b][0] + 1e-9)
            self.assertLessEqual(eff[a][1], eff[b][1] + 1e-9)
        for large, small in eff.values():
            self.assertTrue(1.0 <= large <= 200.0 and 1.0 <= small <= 200.0)

    def test_seeded_and_works_for_many_features(self):
        X, B = np.random.default_rng(2).uniform(size=(3, 36)), np.random.default_rng(3).uniform(size=(100, 36))
        sc = cond_scale(B)
        a = cs.effective_rows_sampled(X, B, sc, 0.2, np.random.default_rng(1))
        b = cs.effective_rows_sampled(X, B, sc, 0.2, np.random.default_rng(1))
        self.assertEqual(a, b)
        self.assertTrue(np.all(np.isfinite(a)))


class TestExplainerCall(unittest.TestCase):
    def test_explain_with_sigma_equals_the_shared_game_table(self):
        rng = np.random.default_rng(4)
        X = rng.uniform(size=(300, 4))
        y = ((X[:, 0] + X[:, 1]) > 1).astype(int)
        model = CFMNN().fit(X, y)
        B, Xte = X[:60], X[100:104]
        scale = cond_scale(B)
        ps = PaceShap(model, B, n_permutations=32, seed=1, sigma=0.2, scale=scale)
        self.assertTrue(ps.exact)
        base = PaceShap(model, B, n_permutations=32, seed=1, scale=scale)
        via_games = base.explain_games(Xte, (0.2,))[0]
        self.assertTrue(np.allclose(ps.explain(Xte), via_games, atol=1e-12))


class TestRunOne(unittest.TestCase):
    def test_a_small_run(self):
        rows = cs.run_one("S1", 120, n=4, settings=[(0.2, 200), (0.4, 200)])
        df = pd.DataFrame(rows)
        self.assertEqual(set(df.method), set(cs.METHODS))
        self.assertEqual(sorted(set(df.setting)), ["sigma=0.2, rows=200", "sigma=0.4, rows=200"])
        self.assertEqual(len(df), 4)
        self.assertTrue((df.eff_rows_large <= df.rows).all() and (df.eff_rows_small <= df.rows).all())
        self.assertTrue(df.off_share.between(0, 1).all())
        a = df[df.setting == "sigma=0.2, rows=200"].eff_rows_large.iloc[0]
        b = df[df.setting == "sigma=0.4, rows=200"].eff_rows_large.iloc[0]
        self.assertLessEqual(a, b)                                    # wider weights, more rows count
        for col in ("cond_margin_auc", "cond_insertion_margin_auc", "margin_auc", "insertion_margin_auc"):
            self.assertTrue(np.isfinite(df[col]).all())
        self.assertEqual(cs.SETTINGS, [(0.05, 200), (0.2, 200), (0.4, 200), (0.2, 800)])


class TestSummary(unittest.TestCase):
    def table(self, shift):
        recs = []
        for ds in ("S1", "magic"):
            for s in range(110, 120):
                for sigma, R in cs.SETTINGS:
                    for m in cs.METHODS:
                        base = 0.4 + 0.001 * s
                        better = shift if m == "PACE-SHAP-cond" else 0.0
                        recs.append({"dataset": ds, "seed": s, "setting": cs.label(sigma, R), "sigma": sigma,
                                     "rows": R, "method": m, "eff_rows_large": 20.0, "eff_rows_small": 50.0,
                                     "off_share": 0.1, "margin_auc": base, "insertion_margin_auc": base,
                                     "cond_margin_auc": base + better + 0.0001 * (s % 3),
                                     "cond_insertion_margin_auc": base - better + 0.0001 * (s % 3),
                                     "gt_top3_precision": np.nan, "ms_per_explanation": 1.0, "n": 50})
        return pd.DataFrame(recs)

    def test_verdicts_for_every_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = cs.summarise(self.table(-0.05), Path(tmp))
            self.assertTrue((Path(tmp) / "summary.md").exists())
        for sigma, R in cs.SETTINGS:
            self.assertIn(f"**{cs.label(sigma, R)}, cond_margin_auc** (negative favours PACE-SHAP-cond): "
                          "better on 2, level on 0, worse on 0 of 2 datasets.", text)
            self.assertIn(f"**{cs.label(sigma, R)}, cond_insertion_margin_auc** (positive favours "
                          "PACE-SHAP-cond): better on 2, level on 0, worse on 0 of 2 datasets.", text)
        with tempfile.TemporaryDirectory() as tmp:
            text = cs.summarise(self.table(0.0), Path(tmp))
        self.assertIn("better on 0, level on 2, worse on 0 of 2 datasets.", text)


if __name__ == "__main__":
    unittest.main()
