"""Tests for scripts/run_fmm_baselines.py (Run P2): the split, theta tuning against the repair
run's, the resume rule, the comparison's Holm step, and one small end-to-end task."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_fmm_baselines as fb  # noqa: E402
import run_repair_theta as rt  # noqa: E402
from pace_cfmnn.fmm_baselines import tune_theta  # noqa: E402
from run_readable_baselines import prepare as paper_prepare  # noqa: E402


class TestSplit(unittest.TestCase):
    def test_paper_split_unless_skin_share(self):
        a, b = fb.prepare("thyroid", 194), paper_prepare("thyroid", 194)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)
        a = fb.prepare("thyroid", 194, 0.2)                     # the share applies to Skin only
        np.testing.assert_array_equal(a[0], b[0])

    def test_skin_share_is_stratified(self):
        full = paper_prepare("skin", 194)
        part = fb.prepare("skin", 194, 0.2)
        n_full, n_part = len(full[1]) + len(full[3]), len(part[1]) + len(part[3])
        self.assertAlmostEqual(n_part / n_full, 0.2, places=3)
        for yf, yp in ((full[1], part[1]), (full[3], part[3])):
            self.assertAlmostEqual(np.mean(yp == yp.max()), np.mean(yf == yf.max()), places=2)
        self.assertTrue((part[0] >= 0).all() and (part[0] <= 1).all())


class TestTuning(unittest.TestCase):
    def test_cfmnn_theta_matches_the_repair_run(self):
        Xtr, ytr, _, _ = paper_prepare("wine", 194)
        theta, _ = tune_theta("CFMNN", Xtr, ytr, 194)
        self.assertEqual(theta, rt.theta_val("wine", 194, Xtr, ytr)[0])


class TestResume(unittest.TestCase):
    def test_pending_is_per_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "runs.csv"
            pd.DataFrame([{"dataset": "skin", "seed": 250, "family": "CFMNN", "x": 1}]).to_csv(csv, index=False)
            tasks = [("skin", 250, "CFMNN", None), ("skin", 250, "EFMNN", None), ("skin", 251, "CFMNN", None)]
            self.assertEqual(fb.pending(csv, tasks, resume=True), tasks[1:])
            with self.assertRaises(SystemExit):
                fb.pending(csv, tasks)


def _fake(gap=1.0):
    rows = []
    for ds in ("thyroid", "magic"):
        for s in range(10):
            for m, v in [(fb.METHOD, 80 + gap * (ds == "thyroid") + 0.01 * s)] + \
                        [(b, 80 + 0.01 * s) for b in fb.BASELINES]:
                rows.append({"dataset": ds, "seed": s, "model": m, "macro_f1": v})
    return pd.DataFrame(rows)


class TestCompare(unittest.TestCase):
    def test_holm_across_all_tests(self):
        res = fb.compare(_fake(1.0), fb.METHOD, fb.BASELINES).set_index(["baseline", "dataset"])
        self.assertEqual(len(res), 8)
        r = res.loc[("EFMNN-val", "thyroid")]
        self.assertEqual((r.wins, r.losses), (10, 0))
        self.assertAlmostEqual(r.p_holm, min(1.0, 8 * 0.001953125))
        self.assertEqual(r.verdict, "better")
        self.assertEqual(res.loc[("EFMNN-val", "magic")].verdict, "no significant difference")
        res = fb.compare(_fake(-1.0), fb.METHOD, fb.BASELINES)
        self.assertEqual(int((res.verdict == "worse").sum()), 4)
        self.assertIn("Mixed", fb.reading(res))


class TestProjection(unittest.TestCase):
    def test_rule(self):
        df = pd.DataFrame([{"dataset": d, "seed": 194, "family": f, "task_s": 3600.0, "model": m}
                           for d in ("magic", "skin") for f in fb.FAMILIES for m in ("a", "b")])
        self.assertAlmostEqual(fb.projection_hours(df, jobs=6), 10 * 10 / 6)        # rows of a task count once
        self.assertIn("so Skin runs in full", fb.projection_text(df, 6))
        self.assertAlmostEqual(fb.projection_hours(df, jobs=6, running=5, elapsed_s=7200), (10 + 10) * 10 / 6)
        self.assertIn("--skin-share 0.2", fb.projection_text(df, 6, running=5, elapsed_s=7200))
        one_long = df.drop_duplicates("family").assign(task_s=[50 * 3600.0] + [3600.0] * 4)
        self.assertAlmostEqual(fb.projection_hours(one_long, n_seeds=1, jobs=6), 50)     # never below the longest task


class TestEndToEnd(unittest.TestCase):
    def test_one_cfmnn_and_one_baseline_task(self):
        rows = fb.run_one("wine", 194, "CFMNN") + fb.run_one("wine", 194, "IOL-GFMM")
        df = pd.DataFrame(rows)
        self.assertEqual(sorted(df.model), sorted(["CFMNN-val", "CFMNN-SR-val", "CFMNN-0.3", "CFMNN-SR-0.3",
                                                   "IOL-GFMM-val", "IOL-GFMM-0.3"]))
        self.assertTrue(df.macro_f1.between(0, 100).all())
        self.assertTrue((df[df.model == "IOL-GFMM-0.3"].theta == 0.3).all())
        sr = df.set_index("model")
        self.assertLessEqual(sr.loc["CFMNN-SR-0.3", "boxes"], sr.loc["CFMNN-0.3", "boxes"])  # shrinks or removes
        with tempfile.TemporaryDirectory() as tmp:
            text = fb.summarise(df.assign(dataset="thyroid"), Path(tmp))
        self.assertIn("## Theta chosen by cross-validation", text)


if __name__ == "__main__":
    unittest.main()
