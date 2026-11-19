"""Run T: the timing check calls each setup as RQ3 did and checks it against RQ3."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import time_repair_arms as tr  # noqa: E402


def problem(seed=0, n=1500, d=4):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    y = ((X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.15, n)) > 0.75).astype(int) + (X[:, 2] > 0.8)
    return X, y


def part():
    X, y = problem(4)
    cut = {"train": (0, 600), "pool": (600, 1000), "inner": (1000, 1100), "accept": (1100, 1300),
           "test": (1300, 1500)}
    return {k: (X[a:b], y[a:b]) for k, (a, b) in cut.items()}


class TestRunT(unittest.TestCase):
    def run_toy(self, ref):
        sr_calls, al_calls = [], []
        real_sr, real_al = tr.run_self_repair, tr.run_refinement

        def spy_sr(*a, **kw):
            out = real_sr(*a, **kw)
            sr_calls.append((kw, out[0]))
            return out

        def spy_al(*a, **kw):
            al_calls.append((a, kw))
            return real_al(*a, **kw)

        with mock.patch.object(tr, "split", lambda name, seed: part()), \
                mock.patch.object(tr, "committed", lambda name, seed: ref), \
                mock.patch.object(tr, "run_self_repair", spy_sr), \
                mock.patch.object(tr, "run_refinement", spy_al):
            rows = tr.run_one("toy", 50)
        return rows, sr_calls, al_calls

    def test_setups_are_called_as_rq3_called_them(self):
        rows, sr_calls, al_calls = self.run_toy({})
        self.assertEqual([r["setup"] for r in rows], [s for s, _ in tr.SETUPS])
        self.assertTrue(all(set(r) == set(tr.COLUMNS) for r in rows))
        budget = min(300, 400 // 4)
        self.assertEqual([kw["fast_gate"] for kw, _ in sr_calls], [False, True])
        for kw, _ in sr_calls:
            self.assertEqual((kw["chooser"], kw["budget"], kw["n_rounds"], kw["seed"],
                              kw["gate_labels_are_new"]), ("volume", budget, 10, 50, False))
        self.assertEqual(len(al_calls), 2)
        for a, kw in al_calls:
            self.assertEqual(a[8:10], ("order_ensemble", "A+B+C'-volume"))
            self.assertEqual((kw["budget"], kw["n_rounds"], kw["seed"]), (budget, 10, 50))
            self.assertNotIn("on_accept", kw)                   # no certificate audit
        self.assertIs(al_calls[1][0][0], sr_calls[1][1])        # SR+AL starts from cached SR
        by = {r["setup"]: r for r in rows}
        self.assertEqual(by["AL"]["labels"], budget + 200)
        self.assertEqual(by["SR+AL"]["labels"], budget + 200)
        self.assertEqual(by["SR, old gate"]["labels"], 0)
        self.assertAlmostEqual(by["SR+AL"]["seconds"],
                               by["SR, cached gate"]["seconds"] + by["SR+AL"]["stage_seconds"])
        self.assertEqual(by["SR, old gate"]["test_macro_f1"], by["SR, cached gate"]["test_macro_f1"])
        self.assertFalse(any(r["matches_committed"] for r in rows))   # nothing to match

    def test_verdict_pass_only_when_every_setup_matches(self):
        rows, _, _ = self.run_toy({})
        ref = {arm: next(r["test_macro_f1"] for r in rows if r["setup"] == s)
               for s, arm in tr.SETUPS}
        rows, _, _ = self.run_toy(ref)
        self.assertTrue(all(r["matches_committed"] for r in rows))
        df = pd.concat([pd.DataFrame(rows).assign(dataset=d) for d in tr.DATASETS])
        with tempfile.TemporaryDirectory() as tmp:
            text = tr.summarise(df, Path(tmp))
            self.assertIn("**Verdict: PASS**", text)
            self.assertIn("## Times as a multiple of SR with the cached gate", text)
            bad = df.copy()
            bad.loc[bad.index[0], "matches_committed"] = False
            self.assertIn("**Verdict: FAIL**", tr.summarise(bad, Path(tmp)))
            self.assertIn("**Verdict: FAIL**", tr.summarise(df.iloc[:-1], Path(tmp)))   # missing

    def test_reproduces_rq3_on_thyroid(self):
        try:
            rows = tr.run_one("thyroid", 50)
        except (FileNotFoundError, OSError) as err:             # data not on this machine
            self.skipTest(str(err))
        self.assertTrue(all(r["matches_committed"] for r in rows), rows)


if __name__ == "__main__":
    unittest.main()
