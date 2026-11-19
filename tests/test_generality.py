"""Run I: tie counts, Simpson's contraction next to CFMNN, the gamma sweep, the exact check."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_generality as rg  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.contraction import contract_all  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.ties import centroid_decided, training_error_ties  # noqa: E402
from tests.test_explain import exact_shapley  # noqa: E402
from tests.test_propositions import own_and_wrong_box  # noqa: E402
from tests.test_refine import problem  # noqa: E402


def brute_ties(model, X, y):
    """Training errors, ties and ties inside both boxes, from the memberships alone."""
    M = model.memberships(X)
    own = model.classes_[model.box_class_][None, :] == np.asarray(y)[:, None]
    pred = model.predict(X)
    wrong = pred != y
    win = model.classes_[model.box_class_][None, :] == pred[:, None]
    top_own = np.where(own, M, -np.inf).max(axis=1)
    top_win = np.where(win, M, -np.inf).max(axis=1)
    tied = wrong & (top_own >= top_win - model.tie_tol)
    both = wrong & (top_own >= 1.0) & (top_win >= 1.0)
    return int(wrong.sum()), int(tied.sum()), int(both.sum())


class TestTies(unittest.TestCase):
    def test_every_cfmnn_training_error_is_a_tie_at_every_gamma(self):
        for gamma in rg.GAMMAS:
            X, y = problem(1, n=400, d=4, classes=3)
            m = CFMNN(theta=0.5, gamma=gamma).fit(X, y)
            t = training_error_ties(m, X, y)
            self.assertGreater(t["train_errors"], 0)
            self.assertEqual(t["train_errors_tied"], t["train_errors"])
            self.assertEqual(t["train_errors_tied_inside_both"], t["train_errors"])
            self.assertEqual((t["train_errors"],) * 3, brute_ties(m, X, y))
            own, other = own_and_wrong_box(m, X, y)
            self.assertTrue(own[m.predict(X) != y].all() and other[m.predict(X) != y].all())

    def test_counts_equal_brute_force_after_contraction(self):
        """After a contraction the share need not be 100%; the count must still be right."""
        below = False
        for seed in range(4):
            X, y = problem(seed, n=400, d=3, classes=3)
            m = CFMNN(theta=0.5).fit(X, y)
            for rule in ("simpson", "gfmm"):
                sim, st, _ = contract_all(m, X, rule=rule)
                t = training_error_ties(sim, X, y)
                self.assertEqual((t["train_errors"], t["train_errors_tied"],
                                  t["train_errors_tied_inside_both"]), brute_ties(sim, X, y))
                self.assertLessEqual(t["train_errors_tied_inside_both"], t["train_errors_tied"])
                below |= t["train_errors_tied"] < t["train_errors"]
        self.assertTrue(below, "expected some contraction to leave an error that is not a tie")

    def test_centroid_decided_equals_brute_force(self):
        X, y = problem(2, n=500, d=4, classes=3)
        m = CFMNN(theta=0.5).fit(X[:350], y[:350])
        Xt, yt = X[350:], y[350:]
        got = centroid_decided(m, Xt, yt)
        M = m.memberships(Xt)
        pred = m.predict(Xt)
        cls = m.classes_[m.box_class_]
        own = np.where(cls[None, :] == pred[:, None], M, -np.inf).max(axis=1)
        other = np.where(cls[None, :] != pred[:, None], M, -np.inf).max(axis=1)
        tied = other >= own - m.tie_tol
        self.assertEqual(got["n"], len(yt))
        self.assertEqual(got["n_centroid_decided"], int(tied.sum()))
        self.assertEqual(got["n_errors"], int((pred != yt).sum()))
        self.assertEqual(got["n_errors_centroid_decided"], int((tied & (pred != yt)).sum()))
        self.assertGreater(got["n_centroid_decided"], 0)


class TestPaceOnTheContractedModel(unittest.TestCase):
    def test_pace_shap_is_exact_on_a_contracted_model(self):
        X, y = problem(3, n=500, d=5, classes=3)
        m = CFMNN(theta=0.5).fit(X, y)
        sim, st, _ = contract_all(m, X)
        self.assertGreater(st["contractions"], 0)
        Xe, B = X[:4], X[100:130]
        phi = PaceShap(sim, B).explain(Xe)
        cls = sim.predict_details(Xe).cls
        brute = np.array([exact_shapley(lambda Z, a=a: xb.margin_score(sim, Z, a), x, B)
                          for x, a in zip(Xe, cls)])
        np.testing.assert_allclose(phi, brute, atol=1e-12)
        np.testing.assert_allclose(phi, xb.exact_shap(sim, Xe, B), atol=1e-12)


class TestScript(unittest.TestCase):
    def test_setup(self):
        self.assertEqual(rg.SEEDS, list(range(150, 160)))
        self.assertEqual(rg.GAMMAS, [2.0, 4.0, 8.0])
        self.assertEqual(len(rg.DATASETS), 13)
        self.assertEqual(rg.MODELS, ["gamma=2", "gamma=4", "gamma=8", "Simpson"])
        self.assertEqual(rg.PRIMARY, [("gamma=2", "gamma=4"), ("gamma=8", "gamma=4")])
        self.assertEqual(rg.EXACT_DATASETS, ["S1", "S2", "S3", "shuttle", "magic", "page_blocks"])

    def test_models_rows(self):
        X, y = problem(4, n=500, d=4, classes=3)
        with mock.patch.object(rg, "prepare", return_value=(X[:400], y[:400], X[400:], y[400:])), \
                mock.patch.object(rg, "THETA", 0.5):
            rows = rg.run_models("toy", 0)
        self.assertEqual([r["model"] for r in rows], rg.MODELS)
        self.assertTrue(all(list(r) == list(rows[0]) for r in rows))
        for r, g in zip(rows[:3], rg.GAMMAS):
            m = CFMNN(theta=0.5, gamma=g).fit(X[:400], y[:400])
            self.assertEqual(r["boxes"], m.n_boxes_)
            self.assertAlmostEqual(r["test_accuracy"], 100 * np.mean(m.predict(X[400:]) == y[400:]))
            self.assertEqual(r["train_errors_tied_inside_both"], r["train_errors"])
            self.assertTrue(np.isnan(r["contractions"]))
        sim, st, _ = contract_all(CFMNN(theta=0.5).fit(X[:400], y[:400]), X[:400])
        self.assertEqual(rows[3]["contractions"], st["contractions"])
        self.assertEqual(rows[3]["boxes"], sim.n_boxes_)
        self.assertAlmostEqual(rows[3]["test_accuracy"], 100 * np.mean(sim.predict(X[400:]) == y[400:]))

    def test_summary(self):
        rng = np.random.default_rng(0)
        rows = []
        for ds in ("wine", "magic"):
            for seed in range(150, 160):
                for model, shift in (("gamma=2", 3.0 if ds == "magic" else 0.0),
                                     ("gamma=4", 0.0), ("gamma=8", 0.0), ("Simpson", -1.0)):
                    sim = model == "Simpson"
                    rows.append({"dataset": ds, "seed": seed, "model": model, "n_train": 100,
                                 "n_test": 50, "boxes": 10, "train_errors": 8,
                                 "train_error_pct": 8.0, "train_errors_tied": 4 if sim else 8,
                                 "train_errors_tied_inside_both": 2 if sim else 8,
                                 "test_accuracy": 80 + shift + rng.normal(0, 0.1),
                                 "test_macro_f1": 75.0, "test_errors": 10,
                                 "test_centroid_decided": 5, "test_errors_centroid_decided": 4,
                                 "contractions": 3 if sim else np.nan,
                                 "boxes_removed": 1 if sim else np.nan,
                                 "rows_evicted": 6 if sim else np.nan})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
            exact = [{"dataset": "magic", "seed": 150, "d": 10, "n": 5, "contractions": 7,
                      "boxes": 20, "max_abs_diff": 2e-16, "max_abs_value": 0.1,
                      "pace_ms_per_explanation": 1.0, "shap_exact_ms_per_explanation": 9.0}]
            pd.DataFrame(exact).to_csv(out / "exact.csv", index=False)
            text = rg.summarise(out)
            self.assertIn("**gamma=2 vs gamma=4**: better 1, level 1, worse 0", text)
            self.assertIn("**gamma=8 vs gamma=4**: better 0, level 2, worse 0", text)
            self.assertIn("are not ties, over every dataset, seed and γ: **0** of 480", text)
            self.assertIn("| magic | 100.0 | 50.0 |", text)        # ties: CFMNN, Simpson
            self.assertIn("| magic | 100.0 | 25.0 |", text)        # ... inside both boxes
            self.assertIn("**Check: PASS**", text)
            exact[0]["max_abs_diff"] = 1e-6
            pd.DataFrame(exact).to_csv(out / "exact.csv", index=False)
            self.assertIn("**Check: FAIL**", rg.summarise(out))


if __name__ == "__main__":
    unittest.main()
