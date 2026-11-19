"""Tests for scripts/run_systems.py (Run G): each system's attributions add up to its own
margin, the lead over a random ranking, the comparison tables and one small end-to-end run."""

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_faithfulness as rf  # noqa: E402
import run_readable_baselines as rb  # noqa: E402
import run_systems as rs  # noqa: E402


def toy(n_classes: int, n: int = 150, d: int = 4, seed: int = 0):
    """Rows in [0, 1]^d whose class depends on the first two features."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    y = np.minimum((n_classes * (0.6 * X[:, 0] + 0.4 * X[:, 1])).astype(int), n_classes - 1)
    return X, y


def weighted_score(model, X, w):
    """The score the attributions explain: the class weights of one point applied to rows X."""
    return model.scores(X) @ w


class AddsUp:
    """Shared check: attributions sum to the point's margin minus the background mean."""

    def check(self, model, attr, X, bg):
        det = model.predict_details(X)
        margin = rf._margin(det, det.cls)
        W = rs.class_weights(det.support_rescaled, det.cls)
        for p in range(len(X)):
            # At the point itself the weighted score is the margin the tests measure.
            self.assertAlmostEqual(float(weighted_score(model, X[p:p + 1], W[p])[0]), margin[p], places=9)
            base = float(weighted_score(model, bg, W[p]).mean())
            self.assertAlmostEqual(float(attr[p].sum()), margin[p] - base, places=7)


class TestEBMTerms(AddsUp, unittest.TestCase):
    def fit(self, n_classes):
        warnings.filterwarnings("ignore")
        X, y = toy(n_classes)
        return rs.EBMScored(rb.make("EBM", None, 0, 1).fit(X, y)), X

    def test_terms_and_intercept_give_the_scores(self):
        for c in (2, 3):
            m, X = self.fit(c)
            icpt = np.atleast_1d(np.asarray(m.ebm.intercept_, float))
            icpt = np.array([0.0, icpt[0]]) if c == 2 else icpt
            np.testing.assert_allclose(m.terms(X[:20]).sum(axis=1) + icpt, m.scores(X[:20]), atol=1e-9)

    def test_two_classes_sums_to_margin_difference(self):
        m, X = self.fit(2)
        self.check(m, rs.explain_ebm(m, X[:25], X[50:]), X[:25], X[50:])

    def test_three_classes_sums_to_margin_difference(self):
        m, X = self.fit(3)
        self.check(m, rs.explain_ebm(m, X[:25], X[50:]), X[:25], X[50:])

    def test_predicted_class_matches_the_library(self):
        for c in (2, 3):
            m, X = self.fit(c)
            np.testing.assert_array_equal(m.ebm.classes_[m.predict_details(X).cls], m.ebm.predict(X))

    def test_rejects_a_model_with_pair_terms(self):
        fake = type("E", (), {"term_features_": [(0,), (1,), (0, 1)]})()
        with self.assertRaises(ValueError):
            rs.EBMScored(fake)


class TestTreeShap(AddsUp, unittest.TestCase):
    def fit(self, n_classes, leaves):
        X, y = toy(n_classes, n=300)
        return rs.TreeScored(rb.make("Tree", leaves, 0, 1).fit(X, y)), X

    def test_values_add_up_per_class(self):
        m, X = self.fit(3, 8)
        sv = rs.tree_shap_values(m.tree, X[:20], X[100:])
        self.assertEqual(sv.shape, (20, X.shape[1], 3))
        np.testing.assert_allclose(sv.sum(axis=1), m.scores(X[:20]) - m.scores(X[100:]).mean(axis=0),
                                   atol=1e-7)

    def test_sums_to_margin_minus_background_mean(self):
        for c, leaves in ((2, 4), (3, 8)):
            m, X = self.fit(c, leaves)
            self.check(m, rs.explain_tree(m, X[:25], X[100:]), X[:25], X[100:])

    def test_pure_leaves_split_the_runner_up_equally(self):
        # A deep tree has pure leaves: every other class scores 0, so they share the -1.
        m, X = self.fit(4, 256)
        det = m.predict_details(X[:25])
        self.assertTrue(np.any(det.support_rescaled.max(axis=1) == 1.0))
        self.check(m, rs.explain_tree(m, X[:25], X[100:]), X[:25], X[100:])


class TestClassWeights(unittest.TestCase):
    def test_runner_up_gets_minus_one(self):
        w = rs.class_weights(np.array([[0.1, 0.7, 0.2]]), np.array([1]))
        np.testing.assert_allclose(w, [[0.0, 1.0, -1.0]])

    def test_tied_runners_up_share_it(self):
        w = rs.class_weights(np.array([[1.0, 0.0, 0.0, 0.0], [0.2, 0.4, 0.4, 0.0]]), np.array([0, 1]))
        np.testing.assert_allclose(w, [[1.0, -1 / 3, -1 / 3, -1 / 3], [0.0, 1.0, -1.0, 0.0]])

    def test_margin_attribution_by_hand(self):
        phi = np.array([[[1.0, 2.0, 4.0], [0.5, 0.0, -1.0]]])        # (1 point, 2 features, 3 classes)
        attr = rs.margin_attribution(phi, np.array([[0.1, 0.7, 0.2]]), np.array([1]))
        np.testing.assert_allclose(attr, [[2.0 - 4.0, 0.0 + 1.0]])


class TestLead(unittest.TestCase):
    def test_deletion_lower_is_better(self):
        self.assertAlmostEqual(rs.lead(0.2, 0.4), 0.5)               # (0.4 - 0.2) / 0.4
        self.assertAlmostEqual(rs.lead(0.5, 0.4), -0.25)             # worse than random

    def test_insertion_higher_is_better(self):
        self.assertAlmostEqual(rs.lead(0.6, 0.4, higher_better=True), 0.5)

    def test_negative_random_keeps_the_sign(self):
        self.assertAlmostEqual(rs.lead(-0.3, -0.2), 0.5)             # (-0.2 + 0.3) / 0.2

    def test_same_lead_at_any_scale(self):
        self.assertAlmostEqual(rs.lead(2.0, 4.0), rs.lead(0.02, 0.04))

    def test_zero_random_is_nan(self):
        self.assertTrue(np.isnan(rs.lead(0.3, 0.0)))


class TestCompare(unittest.TestCase):
    def frame(self):
        rows = []
        for ds, shift in (("S1", 1.0), ("S2", 1.0), ("S3", 1.0), ("wine", -1.0)):
            for seed in range(10):
                rows.append({"dataset": ds, "seed": seed, "model": rs.CFMNN_SYS, "v": shift + 0.01 * seed})
                rows.append({"dataset": ds, "seed": seed, "model": rs.EBM_SYS, "v": 0.0})
        return pd.DataFrame(rows)

    def test_verdicts_and_direction(self):
        res = rs.compare(self.frame(), rs.CFMNN_SYS, rs.EBM_SYS, "v", ["S1", "S2", "S3", "wine"])
        self.assertEqual(list(res.verdict), ["better", "better", "better", "worse"])
        np.testing.assert_allclose(res.p_holm, rb.holm(res.p.to_numpy()))
        low = rs.compare(self.frame(), rs.CFMNN_SYS, rs.EBM_SYS, "v", ["S1", "wine"], lower_better=True)
        self.assertEqual(list(low.verdict), ["worse", "better"])

    def test_synthetic_family_is_one_unit(self):
        u = rs.units(self.frame(), "v")
        self.assertEqual(sorted(u.dataset.unique()), sorted([rs.FAMILY, "wine"]))
        res = rs.compare(u, rs.CFMNN_SYS, rs.EBM_SYS, "v", [rs.FAMILY, "wine"])
        self.assertEqual(list(res.n), [10, 10])
        self.assertAlmostEqual(res["diff"][0], 1.045)


class TestNearZeroRule(unittest.TestCase):
    def test_flagged_dataset_uses_the_gain(self):
        rows = []
        for ds, rnd in (("wine", 0.01), ("thyroid", 0.5)):
            for seed in range(10):
                # The lead says CFMNN is better on both; the gain says worse.
                rows.append({"dataset": ds, "seed": seed, "model": rs.CFMNN_SYS, "random_deletion_auc": rnd,
                             "deletion_lead": 2.0 + 0.01 * seed, "deletion_gain": -1.0 - 0.01 * seed})
                rows.append({"dataset": ds, "seed": seed, "model": rs.EBM_SYS, "random_deletion_auc": 0.5,
                             "deletion_lead": 0.0, "deletion_gain": 0.0})
        df = pd.DataFrame(rows)
        self.assertEqual(rs.flagged(df, rs.CFMNN_SYS, rs.EBM_SYS), {"wine"})
        res = rs.near_zero_rule(df, rs.EBM_SYS, ["wine", "thyroid"])["per dataset"]
        self.assertEqual(list(res.from_gain), [True, False])
        self.assertEqual(list(res.verdict), ["worse", "better"])


class TestSetup(unittest.TestCase):
    def test_study_design(self):
        self.assertEqual(rs.SEEDS, list(range(120, 130)))
        self.assertEqual(len(rs.DATASETS), 11)
        self.assertEqual(rs.SYSTEMS[0], rs.CFMNN_SYS)

    def test_end_to_end_on_wine(self):
        rows = rs.run_one("wine", 99, n_explain=8)
        self.assertEqual([r["model"] for r in rows], rs.SYSTEMS)
        for r in rows:
            self.assertEqual(r["n_explained"], 8)
            self.assertTrue(0 <= r["accuracy"] <= 100)
            self.assertAlmostEqual(r["deletion_lead"],
                                   rs.lead(r["deletion_auc"], r["random_deletion_auc"]))
            self.assertAlmostEqual(r["insertion_lead"],
                                   rs.lead(r["insertion_auc"], r["random_insertion_auc"], True))
        with tempfile.TemporaryDirectory() as tmp:
            text = rs.summarise(pd.DataFrame(rows), Path(tmp))
            self.assertIn("Primary: normalised deletion lead", text)
            self.assertTrue((Path(tmp) / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
