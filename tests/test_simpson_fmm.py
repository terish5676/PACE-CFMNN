"""Run M: Simpson's fuzzy min-max classifier trained from scratch (pace_cfmnn/simpson_fmm.py)."""

import itertools
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.simpson_fmm import SimpsonFMM  # noqa: E402
from pace_cfmnn.ties import centroid_decided, training_error_ties  # noqa: E402
from test_explain import exact_shapley  # noqa: E402


# ------------------------------------------------- a literal, slow transcription
def _f(t, gamma):
    """Simpson's ramp, one number at a time."""
    return max(0.0, 1.0 - max(0.0, gamma * min(1.0, t)))


def _membership(x, v, w, gamma):
    return sum(0.5 * (_f(x[i] - w[i], gamma) + _f(v[i] - x[i], gamma)) for i in range(len(x))) / len(x)


def _case(vj, wj, vk, wk):
    """Simpson's (1992) four overlap cases for one feature, as published."""
    if vj < vk < wj < wk:
        return 1, wj - vk
    if vk < vj < wk < wj:
        return 2, wk - vj
    if vj < vk <= wk < wj:
        return 3, min(wk - vj, wj - vk)
    if vk < vj <= wj < wk:
        return 4, min(wj - vk, wk - vj)
    return 0, None


def naive_simpson(X, y, theta, gamma):
    """Simpson's training, row by row and feature by feature, with CFMNN's size limit
    (sum of sides <= d * theta) and expansion rule (the best box that can still expand)."""
    d = X.shape[1]
    V, W, C = [], [], []
    for x, c in zip(X.tolist(), y.tolist()):
        same = [j for j in range(len(V)) if C[j] == c]
        if not same:
            V.append(list(x)); W.append(list(x)); C.append(c)
            continue
        b = [_membership(x, V[j], W[j], gamma) for j in same]
        best = max(range(len(same)), key=lambda q: (b[q], -q))     # first box on exact ties
        if b[best] >= 1.0:
            continue
        fits = [q for q in range(len(same))
                if sum(max(W[same[q]][i], x[i]) - min(V[same[q]][i], x[i]) for i in range(d)) <= d * theta]
        if not fits:
            V.append(list(x)); W.append(list(x)); C.append(c)
            continue
        q = max(fits, key=lambda q: (b[q], -q))
        j = same[q]
        V[j] = [min(V[j][i], x[i]) for i in range(d)]
        W[j] = [max(W[j][i], x[i]) for i in range(d)]
        for k in range(len(V)):                     # overlap test against every other class
            if C[k] == C[j]:
                continue
            cases = [_case(V[j][i], W[j][i], V[k][i], W[k][i]) for i in range(d)]
            if any(cs == 0 for cs, _ in cases):
                continue
            i = min(range(d), key=lambda i: (cases[i][1], i))   # smallest overlap, lowest feature
            cs = cases[i][0]
            if cs == 1:
                V[k][i] = W[j][i] = 0.5 * (W[j][i] + V[k][i])
            elif cs == 2:
                V[j][i] = W[k][i] = 0.5 * (W[k][i] + V[j][i])
            elif cs == 3:
                if W[k][i] - V[j][i] < W[j][i] - V[k][i]:
                    V[j][i] = W[k][i]
                else:
                    W[j][i] = V[k][i]
            else:
                if W[k][i] - V[j][i] < W[j][i] - V[k][i]:
                    W[k][i] = V[j][i]
                else:
                    V[k][i] = W[j][i]
    return np.array(V), np.array(W), np.array(C)


def overlapping_pairs(model):
    """Pairs of boxes of different classes that Simpson's test says overlap."""
    out = []
    for j, k in itertools.combinations(range(model.n_boxes_), 2):
        if model.box_class_[j] == model.box_class_[k]:
            continue
        cases = [_case(model.V_[j, i], model.W_[j, i], model.V_[k, i], model.W_[k, i])[0]
                 for i in range(model.V_.shape[1])]
        if all(cases):
            out.append((j, k))
    return out


def toy(n=300, d=3, seed=0, classes=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    y = np.digitize(X[:, 0] + 0.6 * X[:, 1] + rng.normal(0, 0.15, n), [0.6, 1.0])[:, None].ravel()
    return X, y % classes


class TestTraining(unittest.TestCase):
    def test_matches_the_literal_transcription(self):
        for seed, (theta, gamma) in enumerate([(0.3, 4.0), (0.15, 4.0), (0.3, 2.0), (0.5, 8.0)]):
            X, y = toy(250, d=3, seed=seed)
            m = SimpsonFMM(theta=theta, gamma=gamma).fit(X, y)
            V, W, C = naive_simpson(X, np.searchsorted(m.classes_, y), theta, gamma)
            self.assertEqual(m.n_boxes_, len(V))
            np.testing.assert_array_equal(m.V_, V)
            np.testing.assert_array_equal(m.W_, W)
            np.testing.assert_array_equal(m.box_class_, C)
            self.assertGreater(m.fit_stats_["n_contractions"], 0)     # the contraction ran

    def test_without_contraction_it_is_cfmnns_expansion(self):
        # one class: no overlap test can fire, so the boxes are CFMNN's (search expansion)
        X, _ = toy(200, seed=7)
        y = np.zeros(len(X), dtype=int)
        s, c = SimpsonFMM().fit(X, y), CFMNN().fit(X, y)
        np.testing.assert_array_equal(s.V_, c.V_)
        np.testing.assert_array_equal(s.W_, c.W_)
        self.assertEqual(s.fit_stats_["n_contractions"], 0)

    def test_an_expansion_into_another_class_is_contracted(self):
        # class 1 grows a box over [0.2, 0.6] x [0.2, 0.6]; class 0 grows one from (0.1, 0.1) to
        # (0.4, 0.4), into it. After the expansion the overlap test cuts them apart.
        X = np.array([[0.2, 0.2], [0.6, 0.6], [0.1, 0.1], [0.4, 0.4]])
        y = np.array([1, 1, 0, 0])
        m = SimpsonFMM(theta=0.5).fit(X, y)
        self.assertEqual((m.n_boxes_, m.fit_stats_["n_contractions"]), (2, 1))
        self.assertEqual(overlapping_pairs(m), [])
        V, W, C = naive_simpson(X, np.searchsorted(m.classes_, y), 0.5, 4.0)
        np.testing.assert_array_equal(m.V_, V)
        np.testing.assert_array_equal(m.W_, W)

    def test_a_new_point_box_is_not_tested(self):
        # Simpson (1992) tests only the box expanded in the last step, and hyperbox-brain's
        # fmnn.py does the same: a point made as a new box inside another class's box stays
        # there. So "no two boxes of different classes overlap after fit" does not hold in
        # general; Run M counts the pairs that still overlap.
        X = np.array([[0.2, 0.2], [0.6, 0.6], [0.4, 0.4]])
        y = np.array([1, 1, 0])
        m = SimpsonFMM(theta=0.5).fit(X, y)
        self.assertEqual((m.n_boxes_, m.fit_stats_["n_contractions"]), (2, 0))
        self.assertEqual(overlapping_pairs(m), [(0, 1)])
        self.assertEqual(SimpsonFMM.overlapping_pairs(m), 1)

    def test_ties_go_to_the_first_box_and_tools_run(self):
        X, y = toy(300, seed=2)
        m = SimpsonFMM().fit(X, y)
        self.assertTrue(np.all(m.centroid_ == 0))
        det = m.predict_details(X)
        M = m.memberships(X)
        top = M.max(axis=1)
        first = np.argmax(M >= top[:, None] - m.tie_tol, axis=1)
        np.testing.assert_array_equal(det.winner, first)
        self.assertEqual(set(training_error_ties(m, X, y)), {"train_errors", "train_error_pct",
                                                            "train_errors_tied", "train_errors_tied_inside_both"})
        self.assertIn("n_centroid_decided", centroid_decided(m, X, y))
        with self.assertRaises(NotImplementedError):
            m.add_points(X[:1], y[:1])


class TestExplanations(unittest.TestCase):
    def test_pace_shap_equals_brute_force_and_exactexplainer(self):
        X, y = toy(300, d=4, seed=5)
        m = SimpsonFMM().fit(X, y)
        B = X[:40]
        pts = X[100:104]
        phi = PaceShap(m, B).explain(pts)
        ref = xb.exact_shap(m, pts, B)
        np.testing.assert_allclose(phi, ref, atol=1e-12)
        for r, x in enumerate(pts):
            a = int(m.predict_details(x[None]).cls[0])
            bf = exact_shapley(lambda Z: xb.margin_score(m, Z, a), x, B)
            np.testing.assert_allclose(phi[r], bf, atol=1e-12)


class TestScript(unittest.TestCase):
    def test_setup(self):
        import run_simpson_fmm as rm
        self.assertEqual(rm.SEEDS, list(range(220, 230)))
        self.assertEqual(rm.EXACT_DATASETS, ["S1", "S2", "S3", "shuttle", "magic", "page_blocks"])
        self.assertEqual((rm.THETA, rm.GAMMA, rm.N_EXACT, rm.EXACT_TOLERANCE), (0.3, 4.0, 5, 1e-9))
        self.assertEqual(rm.MODELS, ["CFMNN", "Simpson-scratch"])
        self.assertEqual(len(rm.DATASETS), 13)

    def test_models_task_and_summary(self):
        import run_simpson_fmm as rm
        rows = rm.run_models("wine", 192)           # a smoke seed, never a study seed
        df = pd.DataFrame(rows)
        self.assertEqual(df.model.tolist(), rm.MODELS)
        self.assertEqual(list(df.columns), rm.COLUMNS)
        c = df[df.model == "CFMNN"].iloc[0]
        s = df[df.model == "Simpson-scratch"].iloc[0]
        self.assertEqual(int(s.overlapping_pairs), SimpsonFMM.overlapping_pairs(
            SimpsonFMM().fit(*rm.prepare("wine", 192)[:2])))
        Xtr, ytr, Xte, yte = rm.prepare("wine", 192)
        self.assertAlmostEqual(c.test_accuracy, 100 * float(np.mean(CFMNN().fit(Xtr, ytr).predict(Xte) == yte)), places=9)
        ex = pd.DataFrame([{"dataset": "S1", "seed": 192, "d": 10, "n": 5, "background_rows": 200,
                            "boxes": 10, "contractions": 3, "max_abs_diff": 1e-16, "max_abs_value": 0.1,
                            "pace_ms_per_explanation": 1.0, "shap_exact_ms_per_explanation": 2.0}])
        with tempfile.TemporaryDirectory() as tmp:
            df.to_csv(Path(tmp) / "runs.csv", index=False)
            ex.to_csv(Path(tmp) / "exact.csv", index=False)
            text = rm.summarise(Path(tmp))
            self.assertIn("**Check: PASS**", text)
            ex.loc[0, "max_abs_diff"] = 1e-6
            ex.to_csv(Path(tmp) / "exact.csv", index=False)
            self.assertIn("**Check: FAIL**", rm.summarise(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
