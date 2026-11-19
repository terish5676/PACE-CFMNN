"""Tests for scripts/run_cannot_hurt.py (Run P11) and the ``on_try`` hook of self-repair: the hook
changes nothing, sees every gated edit, and the check and second-stage rule work as described."""

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_cannot_hurt as ch  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from run_refinement import split  # noqa: E402


class TestHook(unittest.TestCase):
    def test_hook_changes_nothing_and_sees_every_gated_edit(self):
        warnings.filterwarnings("ignore")
        P = split("thyroid", 194)
        Xtr, ytr = P["train"]
        Xt, _ = P["test"]
        model = CFMNN().fit(Xtr, ytr)
        kw = dict(chooser="volume", budget=60, n_rounds=3, seed=194, gate_labels_are_new=False)
        a, _, ea = run_self_repair(model, Xtr, ytr, Xtr, ytr, **kw)
        log = ch.SafetyLog(Xtr, ytr)
        b, _, eb = run_self_repair(model, Xtr, ytr, Xtr, ytr, on_try=log, **kw)
        np.testing.assert_array_equal(a.predict(Xt), b.predict(Xt))
        np.testing.assert_array_equal(a.V_, b.V_)
        gated = [e for e in eb if e["edit"] == "shrink"]
        self.assertEqual(len(log.rows), len(gated))
        self.assertEqual(sum(r["accepted"] for r in log.rows), sum(bool(e["accepted"]) for e in eb))
        self.assertEqual([bool(e["accepted"]) for e in ea], [bool(e["accepted"]) for e in eb])
        for r in log.rows:
            if r["safe"]:
                self.assertEqual(r["train_broken"], 0)


def _fake(safe_rejected=False, broken_safe=0):
    rows = []
    for k in range(10):
        safe = k < 3
        rows.append({"dataset": "page_blocks", "seed": 1, "edit": k, "accepted": not (safe_rejected and k == 0),
                     "safe": safe, "gate_s": 0.1, "n_slab": 0 if safe else 1, "n_won_dropped": 0,
                     "n_above_winner": 0, "same_centroid": safe, "only_this_box": True,
                     "train_broken": broken_safe if k == 1 else (0 if safe else 1), "train_fixed": 1})
    rows.append({"dataset": "thyroid", "seed": 1, "edit": 0, "accepted": True, "safe": False, "gate_s": 0.1,
                 "n_slab": 2, "n_won_dropped": 0, "n_above_winner": 0, "same_centroid": False, "only_this_box": True,
                 "train_broken": 0, "train_fixed": 1})
    return pd.DataFrame(rows)


class TestCheckAndSummary(unittest.TestCase):
    def test_check(self):
        self.assertTrue(ch.check(_fake())[0])
        self.assertFalse(ch.check(_fake(broken_safe=2))[0])

    def test_second_stage_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = ch.summarise(_fake(), Path(tmp))
            self.assertIn("**Second stage (the fast path):** not needed", text)
            row = [line for line in text.splitlines() if line.startswith("| page_blocks |")][0]
            self.assertEqual([c.strip() for c in row.strip("|").split("|")][1:5], ["10", "3", "30.0%", "0"])
            text = ch.summarise(_fake(safe_rejected=True), Path(tmp))
            self.assertIn("**Second stage (the fast path):** run on page_blocks", text)


if __name__ == "__main__":
    unittest.main()
