"""Tests for scripts/run_shap_certificate.py (Run P6): the hook sees every accepted edit and finds no
change outside the certificate on a toy model, values are cached per model, and the check fails as
it should."""

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_shap_certificate as sc  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from run_refinement import split  # noqa: E402
from tests.test_refine import problem  # noqa: E402


class TestHook(unittest.TestCase):
    def test_every_accepted_edit_and_nothing_outside(self):
        warnings.filterwarnings("ignore")
        for name, mode in (("page_blocks", "exact"), ("thyroid", "sampled")):
            P = split(name, 194)
            X, y = P["train"]
            Xt, _ = P["test"]
            model = CFMNN().fit(X, y)
            log = sc.CertLog(Xt[:10], X[:10], seed=0)
            _, _, edits = run_self_repair(model, X, y, X, y, budget=40, n_rounds=2, seed=194,
                                          gate_labels_are_new=False, on_accept=log)
            self.assertEqual(len(log.rows), sum(bool(e["accepted"]) for e in edits))
            self.assertGreater(len(log.rows), 0)
            self.assertTrue(all(r["changed_outside"] == 0 for r in log.rows))
            self.assertEqual({r["mode"] for r in log.rows}, {mode})

    def test_values_are_cached_per_model(self):
        X, y = problem(1, n=300, d=4, classes=2)
        model = CFMNN().fit(X, y)
        log = sc.CertLog(X[:5], X[50:60], seed=0)
        a = log.values(model)
        self.assertIs(log.values(model), a)


def _fake(outside=0):
    return pd.DataFrame([{"dataset": "skin", "seed": 1, "d": 3, "edit": k, "mode": "exact", "n": 100,
                          "certified": 100, "changed": 50, "changed_outside": outside if k == 0 else 0,
                          "max_change_outside": 1e-9 if outside else 0.0, "changed_inside": 50,
                          "rel_change_median": 0.01, "rel_change_max": 0.2} for k in range(4)])


class TestCheckAndSummary(unittest.TestCase):
    def test_check(self):
        self.assertTrue(sc.check(_fake())[0])
        self.assertFalse(sc.check(_fake(outside=2))[0])

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = sc.summarise(_fake(), Path(tmp))
        self.assertIn("**Check: PASS.**", text)
        row = [line for line in text.splitlines() if line.startswith("| skin |")][0]
        self.assertEqual([c.strip() for c in row.strip("|").split("|")][3:7], ["4", "100.0%", "50.0%", "50.0%"])


if __name__ == "__main__":
    unittest.main()
