"""Tests for scripts/run_time_ordered.py (Run P9): the time-ordered stream, the week orders, the
weekly tests, and a short run of every arm on the first weeks (with its summary)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_time_ordered as rto  # noqa: E402
from pace_cfmnn import data  # noqa: E402

HAVE_ELEC = (data.DATA_DIR / "electricity" / data.OPENML_FILES["electricity"]).exists()


@unittest.skipUnless(HAVE_ELEC, "electricity not downloaded")
class TestStream(unittest.TestCase):
    def test_shape_scale_and_time_order(self):
        X, y, names = rto.stream()
        self.assertEqual(X.shape, (134 * 336, 7))
        self.assertEqual(names, ["day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand",
                                 "transfer"])
        self.assertTrue(((X >= 0) & (X <= 1)).all())
        np.testing.assert_allclose(np.unique(X[:, 0]), np.arange(7) / 6)
        # Each day is 48 half-hours in order, so each week holds 7 whole days.
        period = X[:, 1].reshape(-1, 48)
        self.assertTrue(np.all(np.diff(period, axis=1) > 0))


class TestOrders(unittest.TestCase):
    def test_weeks_are_shuffled_inside_only(self):
        orders = rto.week_orders(380, 6)
        for w, o in enumerate(orders):
            np.testing.assert_array_equal(np.sort(o), np.arange(w * 336, (w + 1) * 336))
        again = rto.week_orders(380, 6)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(orders, again)))
        self.assertFalse(np.array_equal(orders[0], rto.week_orders(381, 1)[0]))


class TestWeeklyTests(unittest.TestCase):
    def diffs(self, values):
        idx = pd.MultiIndex.from_tuples([(w, s) for w in range(4, 4 + len(values)) for s in (0, 1)],
                                        names=["week", "seed"])
        return pd.Series(np.repeat(values, 2), index=idx)

    def test_units(self):
        d = self.diffs(np.linspace(1, 2, 12))
        week, seed, month = (rto.test(d, u) for u in ("week", "seed", "month"))
        self.assertEqual((week["n"], seed["n"], month["n"]), (12, 2, 3))
        self.assertEqual(week["a_higher"], 12)
        self.assertLess(week["p"], 0.05)
        self.assertEqual(rto.verdict(week, "p", lower_is_better=True), "worse")
        self.assertEqual(rto.verdict(week, "p", lower_is_better=False), "better")

    def test_no_change(self):
        r = rto.test(self.diffs(np.zeros(8)), "week")
        self.assertTrue(r["same"])
        self.assertEqual(rto.verdict(r, "p", True), "no change")

    def test_settings_match_the_preregistration(self):
        self.assertEqual(rto.SEEDS, list(range(380, 390)))
        self.assertEqual((rto.WEEK, rto.N_WEEKS, rto.START, rto.RETUNE_EVERY), (336, 134, 4, 4))
        self.assertEqual((rto.BUDGET, rto.N_ROUNDS, rto.CHOOSER, rto.FIXED_THETA), (300, 10, "volume", 0.3))


@unittest.skipUnless(HAVE_ELEC, "electricity not downloaded")
class TestShortRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Weeks 4 and 5 are updated and scored; RETUNE refits after week 5 (every 2nd week here).
        cls.rows = rto.run_seed("electricity", 194, n_weeks=7, retune_every=2)
        cls.df = pd.DataFrame(cls.rows)

    def test_every_arm_every_week(self):
        self.assertEqual(len(self.df), 2 * len(rto.ARMS))
        self.assertEqual(set(self.df.week), {4, 5})
        self.assertEqual(set(self.df.arm), set(rto.ARMS))

    def test_certificate_contains_every_change(self):
        cert = self.df[self.df.arm.isin(rto.CERTIFIED)]
        self.assertTrue(cert.cert_contains_changes.all())
        self.assertTrue((cert.cert_pct >= cert.changed_pct - 1e-9).all())
        self.assertTrue(self.df[self.df.arm == "RETUNE"].cert_pct.isna().all())

    def test_counts(self):
        d = self.df
        np.testing.assert_allclose(d.nfr_pct, 100 * d.next_broken / 336)
        self.assertTrue((d.next_broken + d.next_fixed <= d.next_changed).all())
        sr = d[d.arm.isin(["SR", "SR-0.3"])]
        self.assertTrue((sr.sr_accepted <= sr.sr_edits).all())
        self.assertTrue(d[d.arm.isin(["INC", "RETUNE", "INC-0.3"])].sr_edits.isna().all())

    def test_retune_refits_on_schedule(self):
        r = self.df[self.df.arm == "RETUNE"].set_index("week")
        self.assertFalse(r.loc[4, "refit"])
        self.assertTrue(r.loc[5, "refit"])

    def test_retune_equals_inc_when_theta_is_unchanged(self):
        # One-pass training: fit on all rows equals fit on the first rows plus add_points.
        r = self.df[self.df.arm == "RETUNE"].set_index("week")
        i = self.df[self.df.arm == "INC"].set_index("week")
        if r.loc[5, "theta"] == i.loc[5, "theta"]:
            self.assertEqual(r.loc[5, "next_accuracy_after"], i.loc[5, "next_accuracy_after"])
            self.assertEqual(r.loc[5, "boxes"], i.loc[5, "boxes"])

    def test_explanations_are_exact(self):
        sr = self.df[self.df.arm == "SR"]
        self.assertTrue(sr.explain_exact.all())
        self.assertTrue((sr.explain_ms > 0).all())

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.df.to_csv(out / "weeks.csv", index=False)
            text = rto.summarise(out)
        self.assertIn("## Primary: broken predictions in the next week (NFR), SR minus INC", text)
        self.assertIn("| INC | 100.0% |", text)
        self.assertIn("**Containment: PASS**", text)
        self.assertIn("RETUNE's θ at its refits:", text)


if __name__ == "__main__":
    unittest.main()
