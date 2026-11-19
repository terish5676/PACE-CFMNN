"""Run N: Covertype in Run J's sampled-baseline comparison (made-up data, no download)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import run_sampled_baselines as rj  # noqa: E402
import run_sampled_baselines_covertype as rn  # noqa: E402
from pace_cfmnn import data  # noqa: E402
from test_covertype import fake_covertype  # noqa: E402

FAKE = fake_covertype(900, seed=4)


def fake_load(name, *args, **kw):
    if name != "covertype":
        raise AssertionError(name)
    return FAKE


class TestSetup(unittest.TestCase):
    def test_constants_and_budget(self):
        self.assertEqual(rn.SEEDS, [230, 231, 232, 233, 234])
        self.assertEqual(rn.TIMING_SEEDS, [230, 231, 232])
        self.assertEqual((rn.N_EXPLAIN, rn.TIMING_N), (50, 10))
        self.assertEqual(rn.METHODS, rj.METHODS)
        b = rj.budget(200, 54)
        self.assertEqual(b["rows"], 352_000)                    # 200 x 32 x 55
        self.assertEqual((b["perm_max_evals"], b["kernel_coalitions"]), (1760, 1758))

    def test_the_task_is_run_js(self):
        with mock.patch.object(rj, "run_one", return_value=["row"]) as one, \
                mock.patch.object(rj, "time_one", return_value=["t"]) as tim:
            self.assertEqual(rn.run_one("covertype", 230), ["row"])
            self.assertEqual(rn.time_one("covertype", 230), ["t"])
        one.assert_called_once_with("covertype", 230, 50)
        tim.assert_called_once_with("covertype", 230, 10)


class TestTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(data, "load", side_effect=fake_load):
            cls.df = pd.DataFrame(rn.run_one("covertype", 3, n_explain=3))
            cls.t = pd.DataFrame(rn.time_one("covertype", 3, n_explain=2))

    def test_rows(self):
        self.assertEqual(self.df.method.tolist(), rj.METHODS)
        self.assertTrue((self.df.d == 54).all() and (self.df.n == 3).all())
        self.assertTrue((self.df.budget_rows == 352_000).all())
        self.assertTrue((self.df.rows_per_explanation <= self.df.budget_rows).all())
        self.assertTrue((self.df.n_exact == 0).all() and self.df.rel_l1_error.isna().all())   # no exact at 54
        self.assertTrue(np.isfinite(self.df.margin_auc).all())
        self.assertEqual(self.t.method.tolist(), [rj.PACE, rj.PERM, rj.KERNEL])

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = rn.summarise(self.df, Path(tmp))
            self.assertNotIn("measured alone", text)
            self.t.to_csv(Path(tmp) / "timing.csv", index=False)
            text2 = rn.summarise(self.df, Path(tmp))
        self.assertIn("Descriptive: no comparison is tested", text)
        self.assertIn("352,000 model rows per explanation", text)
        for m in rj.METHODS:
            self.assertIn(f"| {m} |", text)
        self.assertIn("Every baseline within the budget: yes", text)
        self.assertIn("measured alone", text2)
        self.assertIn("Times PACE-SHAP's", text2)


if __name__ == "__main__":
    unittest.main()
