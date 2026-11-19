"""Run G, amendment 1: scores on untied points, how the primary reads, and the timing pass."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_faithfulness as rf  # noqa: E402
import run_systems as rs  # noqa: E402


class Stub:
    """Two classes; the margin is x0 - x1, and exactly 0 where x0 == x1 (a tied point)."""

    def predict_details(self, X):
        X = np.asarray(X, float)
        s = np.column_stack([X[:, 0], X[:, 1]])
        return SimpleNamespace(cls=s.argmax(axis=1), support_rescaled=s)


def stub_problem(n_tied=6, n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 4))
    X[:n_tied, 1] = X[:n_tied, 0]
    return Stub(), X, rng.uniform(size=(30, 4)), rng.normal(size=(n, 4))


class TestUntied(unittest.TestCase):
    def test_all_point_values_are_unchanged_and_untied_values_are_the_untied_mean(self):
        m, X, bev, attr = stub_problem()
        out = rs.faithfulness(m, X, attr, bev, 5)
        old = rs.curves(m, X, attr, bev, 5)
        for k, v in old.items():
            self.assertEqual(out[k], v)
        rnd = [rs.curves(m, X, np.random.default_rng(5 + 1000 * (k + 1)).uniform(size=X.shape), bev, 5)
               for k in range(rs.RANDOM_RANKINGS)]
        for k in old:
            self.assertEqual(out["random_" + k], float(np.mean([r[k] for r in rnd])))
        self.assertEqual(out["deletion_lead"], rs.lead(out["deletion_auc"], out["random_deletion_auc"]))

        det = m.predict_details(X)
        untied = rf._margin(det, det.cls) > 0
        self.assertEqual((int(untied.sum()), out["n_untied"]), (14, 14))
        _, dele = rf.deletion_auc(m, X, attr, bev, seed=5, signed=True)
        _, ins = rf.insertion_auc(m, X, attr, bev, seed=5)
        self.assertAlmostEqual(out["deletion_auc_untied"], dele[untied].mean(), places=12)
        self.assertAlmostEqual(out["insertion_auc_untied"], ins[untied].mean(), places=12)
        self.assertNotAlmostEqual(out["deletion_auc_untied"], out["deletion_auc"], places=6)
        self.assertAlmostEqual(out["deletion_lead_untied"],
                               rs.lead(out["deletion_auc_untied"], out["random_deletion_auc_untied"]))
        self.assertAlmostEqual(out["insertion_gain_untied"],
                               out["insertion_auc_untied"] - out["random_insertion_auc_untied"])
        self.assertAlmostEqual(out["deletion_gain_untied"],
                               out["random_deletion_auc_untied"] - out["deletion_auc_untied"])

    def test_no_tied_point_gives_the_all_point_values(self):
        m, X, bev, attr = stub_problem(n_tied=0)
        out = rs.faithfulness(m, X, attr, bev, 5)
        self.assertEqual(out["n_untied"], len(X))
        for k in ("deletion_auc", "insertion_auc", "deletion_lead", "insertion_gain"):
            self.assertAlmostEqual(out[k + "_untied"], out[k], places=12)

    def test_every_point_tied_gives_nan(self):
        m, X, bev, attr = stub_problem(n_tied=20)
        out = rs.faithfulness(m, X, attr, bev, 5)
        self.assertEqual(out["n_untied"], 0)
        self.assertTrue(np.isnan(out["deletion_auc_untied"]) and np.isnan(out["deletion_lead_untied"]))


def frame():
    """wine: no ties. magic: tie-heavy, the lead flips on untied points. letter: tie-heavy,
    too few untied points in one seed."""
    rows = []
    spec = {"wine": (0.0, 50, 1.0, 1.0), "magic": (0.7, 15, 1.0, -1.0), "letter": (0.5, 25, 1.0, 1.0)}
    for ds, (tied, n_untied, lead_all, lead_untied) in spec.items():
        for seed in range(120, 130):
            few = ds == "letter" and seed == 120
            rows.append({"dataset": ds, "seed": seed, "model": rs.CFMNN_SYS, "share_zero_margin": tied,
                         "n_untied": 4 if few else n_untied, "deletion_lead": lead_all + 0.01 * seed,
                         "deletion_lead_untied": lead_untied * (1 + 0.001 * seed),
                         "deletion_gain_untied": lead_untied * (1 + 0.001 * seed),
                         "random_deletion_auc": 0.5, "random_deletion_auc_untied": 0.5,
                         "deletion_gain": 0.3})
            for other in rs.OTHERS:
                rows.append({"dataset": ds, "seed": seed, "model": other, "share_zero_margin": 0.0,
                             "n_untied": 50, "deletion_lead": 0.0, "deletion_lead_untied": 0.0,
                             "deletion_gain_untied": 0.0, "random_deletion_auc": 0.5,
                             "random_deletion_auc_untied": 0.5, "deletion_gain": 0.0})
    return pd.DataFrame(rows)


class TestReading(unittest.TestCase):
    def test_constants(self):
        self.assertEqual((rs.TIE_TOL, rs.MIN_UNTIED, rs.TIE_HEAVY, rs.TIMING_SEEDS),
                         (1e-12, 10, 0.10, [120]))

    def test_eligible_and_tie_heavy_datasets(self):
        df, present = frame(), ["wine", "magic", "letter"]
        self.assertEqual(rs.untied_datasets(df, present), ["wine", "magic"])
        self.assertEqual(rs.tie_heavy(df, present), ["magic", "letter"])

    def test_reading_words(self):
        self.assertEqual(rs.reading("better", "better"), "holds on untied points")
        self.assertEqual(rs.reading("better", "level"), "depends on the tied points")
        self.assertEqual(rs.reading("level", None), "not checked: too few untied points")

    def test_reading_table(self):
        res = rs.untied_reading(frame(), rs.EBM_SYS, ["wine", "magic", "letter"]).set_index("dataset")
        self.assertEqual(list(res.primary), ["better", "better", "better"])
        self.assertEqual(res.loc["wine", "reading"], "not tie-heavy")
        self.assertEqual((res.loc["magic", "untied"], res.loc["magic", "reading"]),
                         ("worse", "depends on the tied points"))
        self.assertEqual((res.loc["letter", "untied"], res.loc["letter", "reading"]),
                         ("not tested", "not checked: too few untied points"))

    def test_summary_sections(self):
        df = frame()
        L = []
        rs.amendment_1(df, L, ["wine", "magic", "letter"], rs.SYSTEMS)
        text = "\n".join(L)
        self.assertIn("## Amendment 1: the primary on untied points", text)
        self.assertIn("| magic | 70.0% | better | worse | depends on the tied points |", text)
        self.assertIn("at least 10 untied points: wine, magic.", text)
        self.assertIn("## Amendment 1: normalised deletion lead, untied points", text)
        old = []
        rs.amendment_1(df.drop(columns=["deletion_lead_untied"]), old, ["wine"], rs.SYSTEMS)
        self.assertEqual(old, [])                      # rows written before the amendment


class TestTimingPass(unittest.TestCase):
    def test_time_one_and_the_timing_table(self):
        rows = rs.time_one("wine", 99, n_explain=6)
        self.assertEqual([r["model"] for r in rows], rs.SYSTEMS)
        self.assertTrue(all(r["ms_per_explanation"] > 0 and r["n_explained"] == 6 for r in rows))
        self.assertEqual(sorted(rows[0]), ["dataset", "fit_s", "model", "ms_per_explanation",
                                           "n_explained", "seed"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            L = []
            rs.timing_table(out, L, ["wine"], rs.SYSTEMS)
            self.assertEqual(L, [])                    # no timing.csv yet: nothing printed
            pd.DataFrame(rows).to_csv(out / "timing.csv", index=False)
            rs.timing_table(out, L, ["wine"], rs.SYSTEMS)
            text = "\n".join(L)
            self.assertIn("measured alone (seeds [99])", text)
            self.assertIn("| wine |", text)

    def test_run_one_has_the_new_columns_after_the_old_ones(self):
        rows = rs.run_one("wine", 99, n_explain=6, systems=[rs.CFMNN_SYS])
        keys = list(rows[0])
        for k in ("n_untied", "deletion_auc_untied", "random_deletion_auc_untied",
                  "deletion_lead_untied", "deletion_gain_untied", "insertion_lead_untied"):
            self.assertIn(k, keys)
        self.assertEqual(rows[0]["n_untied"], 6)       # Wine: no tied point
        self.assertAlmostEqual(rows[0]["deletion_lead_untied"], rows[0]["deletion_lead"], places=12)


if __name__ == "__main__":
    unittest.main()
