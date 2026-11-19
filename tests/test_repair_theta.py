"""Tests for scripts/run_repair_theta.py: the theta lookup, the reproduction check, the tie
count, repair with nothing to repair, the summary's verdicts and one small end-to-end run."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_readable_baselines as rb  # noqa: E402
import run_repair_theta as rt  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.refine import shrink_target  # noqa: E402

HAVE_READABLE = rt.READABLE_RUNS.exists()


@unittest.skipUnless(HAVE_READABLE, "results/readable_baselines/runs.csv is missing")
class TestThetaLookup(unittest.TestCase):
    def test_study_seed_reads_the_readable_run(self):
        r = pd.read_csv(rt.READABLE_RUNS)
        row = r[(r.dataset == "wine") & (r.seed == 70) & (r.model == "CFMNN-val")].iloc[0]
        theta, source = rt.theta_val("wine", 70, None, None)   # no data needed: it is looked up
        self.assertEqual(theta, float(row.setting))
        self.assertEqual(source, "readable_baselines")

    def test_other_seed_is_tuned_the_same_way(self):
        Xtr, ytr, _, _ = rb.prepare("wine", 99)
        theta, source = rt.theta_val("wine", 99, Xtr, ytr)
        expect, _ = rb.tune("CFMNN-val", rb.GRIDS["CFMNN-val"], Xtr, ytr, 99, 1, rb.cv_folds(ytr, 99))
        self.assertEqual(source, "tuned here")
        self.assertEqual(theta, float(expect))

    def test_b0_reproduces_the_readable_run(self):
        # The unrepaired models only: no self-repair result on a study seed is computed here.
        Xtr, ytr, Xte, yte = rb.prepare("wine", 70)
        for model, theta in (("CFMNN-0.3", 0.3), ("CFMNN-val", rt.theta_val("wine", 70, Xtr, ytr)[0])):
            acc = 100 * np.mean(CFMNN(theta=theta, gamma=rt.GAMMA).fit(Xtr, ytr).predict(Xte) == yte)
            self.assertAlmostEqual(acc, rt.readable_row("wine", 70, model).accuracy, places=9)


class TestTrainingErrors(unittest.TestCase):
    def test_counts_match_a_brute_force_recount(self):
        Xtr, ytr, _, _ = rb.prepare("page_blocks", 99)
        m = CFMNN(theta=0.3, gamma=rt.GAMMA).fit(Xtr, ytr)
        got = rt.training_errors(m, Xtr, ytr)
        t = np.searchsorted(m.classes_, ytr)
        err = np.flatnonzero(m.predict(Xtr) != ytr)
        self.assertGreater(len(err), 0)
        self.assertEqual(got["train_errors"], len(err))
        self.assertEqual(got["train_errors_in_winning_box"],
                         sum(shrink_target(m, Xtr[p], int(t[p])) >= 0 for p in err))
        self.assertLessEqual(got["train_errors_own_box_full"], len(err))
        self.assertAlmostEqual(got["train_error_pct"], 100 * len(err) / len(ytr))


class TestRunOne(unittest.TestCase):
    def test_end_to_end_on_a_non_study_seed(self):
        rows = rt.run_one("wine", 99)
        df = pd.DataFrame(rows)
        self.assertEqual(list(df.model), rt.MODELS)
        self.assertEqual(df.set_index("model").loc["B0-0.3", "theta"], 0.3)
        self.assertEqual(df.set_index("model").loc["B0-val", "theta_source"], "tuned here")
        for tag in ("0.3", "val"):
            b0, sr = df.set_index("model").loc[f"B0-{tag}"], df.set_index("model").loc[f"SR-{tag}"]
            self.assertEqual(b0.test_changed, 0)
            self.assertLessEqual(sr.edits_accepted, min(rt.BUDGET, b0.train_errors))
            if b0.train_errors == 0:           # nothing to repair: the repaired model is B0
                self.assertEqual(sr.accuracy, b0.accuracy)
                self.assertEqual(sr.boxes, b0.boxes)
                self.assertEqual(sr.test_changed, 0)
        with tempfile.TemporaryDirectory() as tmp:
            text = rt.summarise(df, Path(tmp))
            self.assertIn("P1 (primary): SR-0.3 vs B0-val", text)
            self.assertTrue((Path(tmp) / "summary.md").exists())

    def test_repair_with_errors_to_fix(self):
        df = pd.DataFrame(rt.run_one("breast_cancer", 99)).set_index("model")
        b0, sr = df.loc["B0-0.3"], df.loc["SR-0.3"]
        self.assertGreater(b0.train_errors, 0)
        self.assertEqual(sr.edits_logged, min(rt.BUDGET, b0.train_errors))   # each error tried once
        self.assertEqual(b0.train_errors_in_winning_box, b0.train_errors)    # every error is a tie
        self.assertLessEqual(sr.train_errors, b0.train_errors)
        self.assertLessEqual(sr.boxes, b0.boxes)                              # shrinking adds no box
        self.assertGreaterEqual(sr.test_changed, sr.test_fixed + sr.test_broken)

    def test_a_b0_that_does_not_reproduce_fails(self):
        fake = pd.Series({"accuracy": -1.0, "macro_f1": -1.0, "setting": 0.3})
        with mock.patch.object(rt, "readable_row", return_value=fake):
            with self.assertRaises(RuntimeError):
                rt.run_one("wine", 99)


class TestSummary(unittest.TestCase):
    def test_verdicts(self):
        rng = np.random.default_rng(0)
        rows = []
        for ds, gain in (("wine", 1.0), ("thyroid", 0.0)):
            for seed in range(70, 80):
                base = 90 + rng.normal()
                acc = {"B0-0.3": base - 2, "SR-0.3": base + gain + rng.normal(0, 0.01) * (gain > 0),
                       "B0-val": base, "SR-val": base}
                for model, a in acc.items():
                    rows.append({"dataset": ds, "seed": seed, "model": model, "theta": 0.3,
                                 "accuracy": a, "macro_f1": a, "boxes": 10, "conditions_per_box": 2.0,
                                 "train_errors": 0, "train_errors_in_winning_box": 0,
                                 "train_errors_own_box_full": 0, "edits_accepted": 0,
                                 "test_fixed": 0, "test_broken": 0, "seconds": 1.0,
                                 "readable_accuracy": np.nan, "readable_macro_f1": np.nan})
        df = pd.DataFrame(rows)
        p1 = rb.compare(df, "SR-0.3", "B0-val").set_index("dataset")
        self.assertEqual(p1.loc["wine", "verdict"], "better")
        self.assertEqual(p1.loc["thyroid", "verdict"], "level")
        self.assertEqual(p1.loc["thyroid", "p"], 1.0)                 # all differences zero
        with tempfile.TemporaryDirectory() as tmp:
            text = rt.summarise(df, Path(tmp))
        self.assertIn("better 1, level 1, worse 0", text)


if __name__ == "__main__":
    unittest.main()
