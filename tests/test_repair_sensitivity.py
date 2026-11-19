"""Run A: the certificates (brute force on random edits), the arms, the gate choice and the
summary's verdicts."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_repair_sensitivity as rs  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.audit import (certify, certify_centroid, explanation_certificate,  # noqa: E402
                              top_groups)
from pace_cfmnn.explain import PaceFlip, decomposition  # noqa: E402
from pace_cfmnn.refine import shrink_box  # noqa: E402


def problem(seed=0, n=500, d=4, classes=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    y = ((X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.15, n)) > 0.75).astype(int)
    if classes == 3:
        y = y + (X[:, 2] > 0.8)
    return X, y


class TestCertificatesNeverMiss(unittest.TestCase):
    """Random edits to small models; the truth is recomputed from scratch before and after."""

    def run_edits(self, seed, theta, n_edits=60, with_add=False):
        X, y = problem(seed)
        model = CFMNN(theta=theta).fit(X, y)
        bg = X[:60]
        values = PaceFlip(model, bg).values_
        lab = X.copy()                                    # the rows that members_ index
        rng = np.random.default_rng(seed + 100)
        # training rows have membership 1 in several boxes: many ties
        rows = np.vstack([X[:150], rng.uniform(size=(60, X.shape[1]))])
        stats = {k: 0 for k in ("edits", "pred", "flip", "dec", "tight", "centroid", "loose",
                                "flipset", "decset", "removed")}
        for step in range(n_edits):
            before = copy.deepcopy(model)
            removed = None
            if with_add and step % 20 == 19:
                p = int(rng.integers(len(X)))
                new = X[p:p + 1] + 0.01
                model.add_points(new, y[p:p + 1])          # member ids continue after the last row
                lab = np.vstack([lab, new])
            else:
                k = int(rng.integers(model.n_boxes_))
                p = int(rng.choice(model.members_[k]))
                info = shrink_box(model, k, lab[p], int(rng.integers(X.shape[1])), lab)
                removed = k if info["removed"] else None
            tb, pb = top_groups(before, rows)
            ta, pa = top_groups(model, rows)
            loose, tight = certify(before, model, tb, ta, removed)
            cent = certify_centroid(before, model, rows, removed)
            flip, dec = explanation_certificate(before, model, rows, values, removed)
            changed = pb != pa
            e0, e1 = PaceFlip(before, bg).explain(rows), PaceFlip(model, bg).explain(rows)
            fch = np.abs(e0 - e1).max(axis=1) > 1e-12
            dch = np.abs(decomposition(before, rows, bg).phi - decomposition(model, rows, bg).phi
                         ).max(axis=1) > 1e-12
            self.assertFalse((changed & ~loose).any(), "loose missed")
            self.assertFalse((changed & ~tight).any(), "tight missed")
            self.assertFalse((changed & ~cent).any(), "centroid missed")
            self.assertFalse((fch & ~flip).any(), "PACE-flip certificate missed")
            self.assertFalse((dch & ~dec).any(), "PACE-decomposition certificate missed")
            self.assertTrue((cent <= tight).all() and (tight <= loose).all())   # nested
            self.assertTrue((dec <= flip).all())
            stats["edits"] += 1
            stats["removed"] += removed is not None
            for name, v in (("pred", changed), ("flip", fch), ("dec", dch), ("tight", tight),
                            ("centroid", cent), ("loose", loose), ("flipset", flip), ("decset", dec)):
                stats[name] += int(v.sum())
        return stats

    def test_never_misses_on_shrinks_removals_and_adds(self):
        total = {}
        for seed, theta in ((1, 0.3), (2, 0.15), (3, 0.5)):
            s = self.run_edits(seed, theta, with_add=True)
            for k, v in s.items():
                total[k] = total.get(k, 0) + v
        self.assertGreater(total["removed"], 0)                    # both kinds of edit happened
        self.assertGreater(total["pred"], 0)                       # edits really change predictions
        self.assertGreater(total["flip"], 0)
        self.assertLessEqual(total["centroid"], total["tight"])
        self.assertLess(total["flipset"], 213 * total["edits"])    # and the certificate is not "all rows"

    def test_certificates_are_selective(self):
        s = self.run_edits(4, 0.3, n_edits=40)
        n = 210 * s["edits"]
        self.assertLess(s["loose"], n)
        self.assertLess(s["flipset"], n)
        self.assertLessEqual(s["decset"], s["flipset"])


class TestGateChoice(unittest.TestCase):
    def test_fast_gate_only_after_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "summary.md"
            self.assertFalse(rs.use_fast_gate(p))                 # no summary: standard gate
            p.write_text("# E\n\n**Verdict: FAIL**\n", encoding="utf-8")
            self.assertFalse(rs.use_fast_gate(p))
            p.write_text("# E\n\n**Verdict: PASS**\n", encoding="utf-8")
            self.assertTrue(rs.use_fast_gate(p))


class TestArms(unittest.TestCase):
    def test_arms_are_built_as_specified(self):
        X, y = problem(9, n=1100)
        part = {"train": (X[:500], y[:500]), "pool": (X[500:900], y[500:900]),
                "test": (X[900:], y[900:])}
        calls = []
        real = rs.run_self_repair

        def spy(*a, **kw):
            calls.append(kw)
            return real(*a, **kw)

        with mock.patch.object(rs, "split", lambda name, seed: part), \
                mock.patch.object(rs, "run_self_repair", spy), \
                mock.patch.object(rs, "use_fast_gate", lambda: True):
            rows = rs.run_one("toy", 111)
        self.assertEqual([r["arm"] for r in rows], rs.MODELS)
        self.assertEqual(len(calls), 8)
        by = dict(zip([r["arm"] for r in rows][1:], calls))
        d = min(300, 400 // 4)
        self.assertEqual([by[a]["budget"] for a in ("SR-default", "SR-b100", "SR-b1000", "SR-ball")],
                         [d, 100, 1000, 500])
        for a in ("SR-gate-none", "SR-gate-strict", "SR-gate-loose", "SR-gate-cumulative"):
            self.assertEqual(by[a]["budget"], d)
        self.assertEqual((by["SR-default"]["max_f1_drop"], by["SR-default"]["max_recall_drop"]), (0.5, 1.0))
        self.assertEqual((by["SR-gate-none"]["max_f1_drop"], by["SR-gate-none"]["max_recall_drop"]),
                         (float("inf"), float("inf")))
        self.assertEqual((by["SR-gate-strict"]["max_f1_drop"], by["SR-gate-strict"]["max_recall_drop"]), (0.0, 0.0))
        self.assertEqual((by["SR-gate-loose"]["max_f1_drop"], by["SR-gate-loose"]["max_recall_drop"]), (1.0, 2.0))
        self.assertEqual({a: kw["gate_reference"] for a, kw in by.items()},
                         {a: ("start" if a == "SR-gate-cumulative" else "last") for a in by})
        for kw in calls:                                           # RQ3's call, and the same seed
            self.assertEqual((kw["chooser"], kw["n_rounds"], kw["seed"], kw["gate_labels_are_new"],
                              kw["fast_gate"]), ("volume", 10, 111, False, True))
        self.assertTrue(all(kw["on_accept"] is None for a, kw in by.items() if a != "SR-default"))
        self.assertIsNotNone(by["SR-default"]["on_accept"])
        # one schema, certificates only on SR-default
        self.assertTrue(all(list(r) == rs.COLUMNS or set(r) == set(rs.COLUMNS) for r in rows))
        cert_rows = [r["arm"] for r in rows if not np.isnan(r["cert_edits"])]
        self.assertEqual(cert_rows, ["SR-default"])
        b0 = rows[0]
        self.assertEqual(b0["boxes_end"], b0["boxes_start"])
        self.assertEqual(b0["churn_changed"], 0)
        # every arm starts from the same model, so B0's start numbers are shared
        self.assertEqual({r["boxes_start"] for r in rows}, {b0["boxes_start"]})
        # an arm that gates nothing accepts at least as many edits as the strict one
        acc = {r["arm"]: r["edits_accepted"] for r in rows}
        self.assertGreaterEqual(acc["SR-gate-none"], acc["SR-gate-strict"])
        self.assertGreaterEqual(acc["SR-ball"], acc["SR-default"])
        self.assertEqual(rows[1]["cert_centroid_misses"], 0)


class TestSummary(unittest.TestCase):
    def table(self, shift_b100=0.0, shift_b1000=0.0, centroid_misses=0):
        recs = []
        for s in range(110, 120):
            for arm in rs.MODELS:
                rec = {c: 0.0 for c in rs.COLUMNS}
                rec.update({"dataset": "thyroid", "seed": s, "arm": arm, "group": "g",
                            "gate_impl": "cached", "gate_reference": "last"})
                base = 50.0 + 0.01 * s
                rec["test_macro_f1"] = base + {"SR-b100": shift_b100 * (s - 109),
                                               "SR-b1000": shift_b1000 * (s - 109)}.get(arm, 0.0)
                if arm == "SR-default":
                    rec["cert_centroid_misses"] = centroid_misses
                else:
                    rec.update({f"cert_{c}_{m}": np.nan for c in rs.CERTS
                                for m in ("misses", "size_share", "precision")})
                    rec["cert_edits"] = np.nan
                recs.append(rec)
        return pd.DataFrame(recs)

    def test_verdicts(self):
        df = self.table(shift_b100=0.5, shift_b1000=-0.5)
        res = rs.primaries(df)
        self.assertEqual(res["SR-b100"].verdict.tolist(), ["better"])
        self.assertEqual(res["SR-b1000"].verdict.tolist(), ["worse"])
        self.assertEqual(res["SR-ball"].verdict.tolist(), ["level"])       # identical to SR-default
        self.assertAlmostEqual(res["SR-b100"]["diff"].iloc[0], 0.5 * 5.5)

    def test_summary_text_and_the_dropped_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = rs.summarise(self.table(0.5, -0.5), Path(tmp))
            self.assertTrue((Path(tmp) / "summary.md").exists())
            self.assertIn("**SR-b100 vs SR-default**: better 1, level 0, worse 0", text)
            self.assertIn("**SR-b1000 vs SR-default**: better 0, level 0, worse 1", text)
            self.assertNotIn("is dropped", text)
            text = rs.summarise(self.table(centroid_misses=1), Path(tmp))
            self.assertIn("The centroid certificate missed, so it is dropped.", text)


if __name__ == "__main__":
    unittest.main()
