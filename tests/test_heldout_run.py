"""Tests for scripts/run_heldout.py (Run P10): the paired comparison and its verdicts, one RQ2
task on a small dataset, the like-for-like check on a held-out dataset, and the summary."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_heldout as rh  # noqa: E402
import run_readable_baselines as rb  # noqa: E402
from pace_cfmnn import data  # noqa: E402


def paired(a_vals, b_vals, ds="toy", key="model", a="A", b="B", metric="m"):
    rows = [{"dataset": ds, "seed": s, key: a, metric: x} for s, x in enumerate(a_vals)]
    rows += [{"dataset": ds, "seed": s, key: b, metric: x} for s, x in enumerate(b_vals)]
    return pd.DataFrame(rows)


class TestCompare(unittest.TestCase):
    def test_verdicts(self):
        base = np.arange(10.0)
        df = pd.concat([paired(base + 1 + 0.01 * base, base, ds="up"),
                        paired(base, base, ds="same"),
                        paired(base + np.r_[1, -1] .repeat(5) * 0.1, base, ds="mixed")])
        res = rh.compare(df, "model", "A", "B", "m", lower_is_better=False).set_index("dataset")
        self.assertEqual(res.loc["up", "verdict"], "better")
        self.assertEqual(res.loc["same", "verdict"], "no change")
        self.assertEqual(res.loc["same", "p"], 1.0)
        self.assertEqual(res.loc["mixed", "verdict"], "no significant difference")
        self.assertEqual((res.loc["up", "seeds_a_better"], res.loc["up", "seeds_a_worse"]), (10, 0))
        low = rh.compare(df, "model", "A", "B", "m", lower_is_better=True).set_index("dataset")
        self.assertEqual(low.loc["up", "verdict"], "worse")
        self.assertEqual(low.loc["up", "seeds_a_worse"], 10)

    def test_holm_matches_the_paper_runs(self):
        p = np.array([0.01, 0.04, 0.03, 0.5])
        np.testing.assert_allclose(rh.holm(p), rb.holm(p))

    def test_datasets_and_seeds(self):
        self.assertEqual(rh.DATASETS, list(data.HELDOUT))
        self.assertEqual(rh.RQ1_SEEDS, list(range(360, 370)))
        self.assertEqual(rh.RQ2_SEEDS, list(range(370, 380)))


class TestRq2Task(unittest.TestCase):
    def test_three_models_counted_against_b0(self):
        wine = rb.prepare("wine", 99)
        with mock.patch.object(rh, "prepare", return_value=wine), \
                mock.patch.object(rh, "theta_val", return_value=(0.2, "tuned here")):
            rows = rh.run_rq2("wine", 99)
        by = {r["model"]: r for r in rows}
        self.assertEqual(set(by), {"B0-0.3", "SR-0.3", "B0-val"})
        self.assertEqual(by["B0-0.3"]["test_changed"], 0)            # compared with itself
        self.assertEqual(by["B0-val"]["theta"], 0.2)
        self.assertEqual(by["B0-val"]["theta_source"], "tuned here")
        self.assertEqual(by["SR-0.3"]["theta"], 0.3)
        self.assertLessEqual(by["SR-0.3"]["edits_accepted"], by["SR-0.3"]["edits_logged"])
        for r in rows:
            self.assertAlmostEqual(r["nfr_pct"], 100 * r["test_broken"] / r["n_test"])
            self.assertLessEqual(r["test_broken"] + r["test_fixed"], r["test_changed"])

    def test_b0_matches_a_plain_fit(self):
        from pace_cfmnn import CFMNN
        Xtr, ytr, Xte, yte = wine = rb.prepare("wine", 98)
        with mock.patch.object(rh, "prepare", return_value=wine), \
                mock.patch.object(rh, "theta_val", return_value=(0.3, "tuned here")):
            rows = rh.run_rq2("wine", 98)
        acc = 100 * np.mean(CFMNN(theta=0.3, gamma=rb.GAMMA).fit(Xtr, ytr).predict(Xte) == yte)
        self.assertAlmostEqual(rows[0]["accuracy"], acc)
        self.assertEqual(rows[2]["test_changed"], 0)                 # theta-val = 0.3: same model


HAVE_MFEAT = (data.DATA_DIR / "mfeat_morphological" / data.OPENML_FILES["mfeat_morphological"]).exists()


@unittest.skipUnless(HAVE_MFEAT, "mfeat_morphological not downloaded")
class TestLikeForLike(unittest.TestCase):
    def test_kernel_shap_is_exact_at_six_features(self):
        # 2^6 - 2 = 62 coalitions < 200: KernelSHAP enumerates them all, so it equals PACE-SHAP.
        check = rh.like_for_like_check("mfeat_morphological", 194, 3)
        self.assertEqual(check["d"], 6)
        self.assertTrue(check["kernel_enumerates"])
        self.assertLess(check["like_for_like_max_abs_diff"], 1e-9)


class TestSummary(unittest.TestCase):
    def test_summary_reads_both_files(self):
        rng = np.random.default_rng(0)
        r1 = []
        for ds in ("phoneme", "wilt"):
            for s in (360, 361, 362):
                for proto in ("cond", "v3"):
                    for m in ("PACE-SHAP", "KernelSHAP"):
                        r1.append({"dataset": ds, "seed": s, "method": m, "protocol": proto,
                                   "margin_auc": rng.uniform(), "insertion_margin_auc": rng.uniform(),
                                   "ms_per_explanation": 1.0 if m == "PACE-SHAP" else 5.0, "d": 5,
                                   "kernel_enumerates": True, "like_for_like_max_abs_diff": 1e-12})
        r2 = []
        for ds in ("phoneme", "wilt"):
            for s in (370, 371, 372):
                for m, f1, nfr in (("B0-0.3", 80, 0), ("SR-0.3", 81, 0.1), ("B0-val", 82, 1.0)):
                    r2.append({"dataset": ds, "seed": s, "model": m, "theta": 0.3 if m != "B0-val" else 0.1,
                               "macro_f1": f1 + 0.01 * s, "nfr_pct": nfr, "boxes": 10,
                               "train_error_pct": 1.0,
                               "test_changed": 0 if (m == "B0-0.3" or ds == "wilt") else 3})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pd.DataFrame(r1).to_csv(out / "rq1_runs.csv", index=False)
            pd.DataFrame(r2).to_csv(out / "rq2_runs.csv", index=False)
            text = rh.summarise(out)
            self.assertTrue((out / "summary.md").exists())
        self.assertIn("| phoneme | 5 | yes | 1.00e-12 |", text)
        self.assertIn("**Check: PASS**", text)
        self.assertIn("5.0×", text)
        self.assertIn("Primary (a)", text)
        self.assertIn("self-repair better on 0, worse on 0, no significant difference on 2", text)
        self.assertIn("0.1 (3)", text)
        self.assertIn("| wilt | 1.00 | 0 of 3 |", text)
        self.assertIn("(it cannot break one): wilt. Counting only the other 1:", text)
        self.assertIn("Secondary: time per explanation", text)

    def test_check_fails_when_the_explainers_differ(self):
        rows = [{"dataset": "wilt", "seed": s, "method": m, "protocol": p, "margin_auc": 0.5,
                 "insertion_margin_auc": 0.5, "ms_per_explanation": 1.0, "d": 5,
                 "kernel_enumerates": True, "like_for_like_max_abs_diff": 1e-6 if s == 361 else 0.0}
                for s in (360, 361) for m in ("PACE-SHAP", "KernelSHAP") for p in ("cond", "v3")]
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(rows).to_csv(Path(tmp) / "rq1_runs.csv", index=False)
            self.assertIn("**Check: FAIL**", rh.summarise(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
