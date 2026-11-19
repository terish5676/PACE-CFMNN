"""Tests for scripts/run_theta_churn.py (Run O): the flip rates, the grid lookup, the
reproduction check, the verdict for a lower-is-better measure, the boxes needed to match repair,
and one small end-to-end run on a smoke seed. No flip between two study-seed models is computed
here: the run's primary is not looked at before the run."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_readable_baselines as rb  # noqa: E402
import run_theta_churn as tc  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

HAVE_READABLE = Path(tc.ROOT / "results" / "readable_baselines" / "runs.csv").exists()


class TestFlipRates(unittest.TestCase):
    def test_counts_and_percentages(self):
        y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        before = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 0])      # wrong at 1 and 3
        after = np.array([0, 0, 0, 1, 2, 1, 0, 1, 2, 0])       # fixes 1 and 3, breaks 2 and 5
        r = tc.flip_rates(y, before, after)
        self.assertEqual((r["test_changed"], r["test_fixed"], r["test_broken"]), (4, 2, 2))
        self.assertAlmostEqual(r["changed_pct"], 40.0)
        self.assertAlmostEqual(r["pfr"], 20.0)
        self.assertAlmostEqual(r["nfr"], 20.0)
        self.assertAlmostEqual(r["fix_share"], 50.0)

    def test_no_change_gives_zero(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0, 0, 1, 1])
        r = tc.flip_rates(y, p, p)
        self.assertEqual((r["test_changed"], r["pfr"], r["nfr"]), (0, 0.0, 0.0))
        self.assertTrue(np.isnan(r["fix_share"]))

    def test_a_change_between_two_wrong_classes_is_neither_fix_nor_break(self):
        r = tc.flip_rates(np.array([0]), np.array([1]), np.array([2]))
        self.assertEqual((r["test_changed"], r["test_fixed"], r["test_broken"]), (1, 0, 0))


class TestGridTheta(unittest.TestCase):
    def test_finds_each_grid_value(self):
        for t in rb.THETAS:
            self.assertEqual(tc.grid_theta(float(str(t))), t)

    def test_rejects_a_value_off_the_grid(self):
        with self.assertRaises(ValueError):
            tc.grid_theta(0.25)


@unittest.skipUnless(HAVE_READABLE, "results/readable_baselines/runs.csv is missing")
class TestReproduction(unittest.TestCase):
    def test_b0_at_03_reproduces_and_a_wrong_value_raises(self):
        # B0-0.3 on Wine, seed 70: its accuracy is already committed by the readable-baselines run.
        Xtr, ytr, Xte, yte = rb.prepare("wine", 70)
        pred = CFMNN(theta=0.3, gamma=rb.GAMMA).fit(Xtr, ytr).predict(Xte)
        acc = 100 * float(np.mean(pred == yte))
        f1 = 100 * float(f1_score(yte, pred, average="macro"))
        ref = tc.check_reproduces("wine", 70, "CFMNN-0.3", acc, f1)
        self.assertIsNotNone(ref)
        with self.assertRaises(RuntimeError):
            tc.check_reproduces("wine", 70, "CFMNN-0.3", acc + 0.1, f1)

    def test_a_seed_without_a_readable_row_is_not_checked(self):
        self.assertIsNone(tc.check_reproduces("wine", 193, "CFMNN-0.3", 0.0, 0.0))


def _fake_runs():
    """Two datasets, 10 seeds; B0 rows at 0.3 and at theta-val, and repair rows."""
    rng = np.random.default_rng(0)
    rows, rep = [], []
    for ds in ("wine", "thyroid"):
        for s in range(10):
            for th, acc, boxes, nfr in ((0.3, 90.0, 50, 0.0), (0.2, 92.0, 80, 2.0), (0.05, 95.0, 300, 3.0)):
                rows.append({"dataset": ds, "seed": s, "theta": th, "is_val": th == 0.05,
                             "is_fixed": th == 0.3, "accuracy": acc + rng.normal(0, 0.1),
                             "macro_f1": acc - 1.0, "boxes": boxes, "nfr": nfr + s / 100,
                             "pfr": 1.0, "changed_pct": nfr + 1.0, "fix_share": 50.0,
                             "train_errors": 10.0, "fit_s": 0.1, "predict_s": 0.1, "task_s": 0.5,
                             "readable_accuracy": np.nan, "readable_macro_f1": np.nan})
            # Repair: low NFR on wine (better), the same NFR as re-tuning on thyroid (level).
            nfr_sr = 0.1 if ds == "wine" else 3.0 + s / 100
            rep.append({"dataset": ds, "seed": s, "model": "SR-0.3", "accuracy": 91.0, "macro_f1": 90.0,
                        "boxes": 50, "test_changed": 5, "test_fixed": 4, "n_test": 100,
                        "test_broken": nfr_sr})
            rep.append({"dataset": ds, "seed": s, "model": "B0-0.3", "accuracy": 90.0, "macro_f1": 89.0,
                        "boxes": 50, "test_changed": 0, "test_fixed": 0, "test_broken": 0, "n_test": 100})
    return pd.DataFrame(rows), pd.DataFrame(rep)


class TestSummaryPieces(unittest.TestCase):
    def test_lower_is_better_verdicts(self):
        df, rep = _fake_runs()
        res = tc.compare_lower(tc.p1_frame(df, rep), "SR-0.3", "B0-val", "nfr").set_index("dataset")
        self.assertEqual(res.loc["wine", "verdict"], "better")       # repair breaks fewer
        self.assertEqual(res.loc["thyroid", "verdict"], "level")     # the same rate
        self.assertLess(res.loc["wine", "diff"], 0)

    def test_boxes_needed_to_match_repair(self):
        df, rep = _fake_runs()
        bm = tc.boxes_to_match(df, rep).set_index("dataset")
        # SR-0.3's accuracy is 91: theta = 0.2 (92%, 80 boxes) is the cheapest B0 that reaches it.
        self.assertEqual(bm.loc["wine", "theta"], 0.2)
        self.assertAlmostEqual(bm.loc["wine", "boxes_vs_0.3"], 80 / 50)

    def test_not_reached(self):
        df, rep = _fake_runs()
        rep.loc[rep.model == "SR-0.3", "accuracy"] = 99.0
        bm = tc.boxes_to_match(df, rep)
        self.assertTrue(bm.theta.isna().all())

    def test_summary_writes_the_primary_table(self):
        df, rep = _fake_runs()
        with tempfile.TemporaryDirectory() as tmp:
            text = tc.summarise(df, Path(tmp), repair=rep)
        self.assertIn("## P1 (primary)", text)
        self.assertIn("better 1, level 1, worse 0", text)
        self.assertIn("| wine | 0.100 |", text)


class TestEndToEnd(unittest.TestCase):
    def test_one_small_task_on_a_smoke_seed(self):
        rows = tc.run_one("wine", 193)
        df = pd.DataFrame(rows)
        self.assertEqual(sorted(df.theta), sorted(rb.THETAS))
        self.assertEqual(int(df.is_val.sum()), 1)
        self.assertEqual(int(df.is_fixed.sum()), 1)
        base = df[df.theta == 0.3].iloc[0]
        self.assertEqual((base.test_changed, base.nfr, base.pfr), (0, 0.0, 0.0))
        self.assertTrue((df.test_fixed + df.test_broken <= df.test_changed).all())
        with tempfile.TemporaryDirectory() as tmp:
            text = tc.summarise(df, Path(tmp), repair=pd.DataFrame(columns=["dataset", "seed", "model"]))
            self.assertIn("P1 is not computed", text)
            self.assertTrue((Path(tmp) / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
