"""Tests for PACE explanations (blueprint 11.2: explain, counterfactual, global)."""

import itertools
import math
import unittest

import numpy as np

from pace_cfmnn import CFMNN, data
from pace_cfmnn.cfmnn_core import feature_terms
from pace_cfmnn.counterfactual import counterfactual
from pace_cfmnn.explain import PaceFlip, contrastive, contrastive_multi, decomposition, pace_flip
from pace_cfmnn import global_importance as gi
from tests.reference_pace_flip import pace_flip_loop


def fitted(seed=0, n=800, d=5):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    y = ((X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.15, n)) > 0.75).astype(int)
    return CFMNN().fit(X, y), X, y


def exact_shapley(f, x, background):
    """Brute-force interventional Shapley values of f at x (small d only)."""
    d = len(x)
    phi = np.zeros(d)

    def v(S):
        Z = background.copy()
        Z[:, list(S)] = x[list(S)]
        return f(Z).mean()

    for i in range(d):
        rest = [k for k in range(d) if k != i]
        for r in range(d):
            for S in itertools.combinations(rest, r):
                w = math.factorial(r) * math.factorial(d - r - 1) / math.factorial(d)
                phi[i] += w * (v(S + (i,)) - v(S))
    return phi


class TestDecomposition(unittest.TestCase):
    def test_sums_to_membership_minus_base(self):
        m, X, _ = fitted()
        dec = decomposition(m, X[:200], X[200:])
        np.testing.assert_allclose(dec.phi.sum(axis=1), dec.membership - dec.base, atol=1e-9)

    def test_equals_exact_shapley_of_box_membership(self):
        m, X, _ = fitted(d=4)
        bg = X[400:460]
        dec = decomposition(m, X[:5], bg)
        for r in range(5):
            j = dec.box[r]
            f = lambda Z: feature_terms(Z, m.V_[j], m.W_[j], m.gamma).mean(axis=1)
            np.testing.assert_allclose(dec.phi[r], exact_shapley(f, X[r], bg), atol=1e-9)


class TestContrastive(unittest.TestCase):
    def test_identities_both_modes(self):
        m, X, _ = fitted(1)
        c = contrastive(m, X)
        det = m.predict_details(X)
        rows = np.arange(len(X))
        mem = c.mode == "membership"
        b_a, b_b = det.support[rows, c.cls_a], m.memberships(X)[rows, np.maximum(c.box_b, 0)]
        np.testing.assert_allclose(c.e[mem].sum(axis=1), (b_a - b_b)[mem], atol=1e-9)
        tie = c.mode == "tie"
        self.assertGreater(tie.sum(), 0)  # training points inside overlapping boxes
        dA = ((X - m.centroid_[c.box_a]) ** 2).sum(axis=1)
        dB = ((X - m.centroid_[np.maximum(c.box_b, 0)]) ** 2).sum(axis=1)
        np.testing.assert_allclose(c.e[tie].sum(axis=1), (dB - dA)[tie], atol=1e-9)
        # The prediction is A, so the gap favours A (>= 0) in both modes.
        self.assertTrue((c.gap[c.mode != "none"] >= -1e-9).all())

    def test_not_uniform_inside_boxes(self):
        m, X, _ = fitted(2)
        c = contrastive(m, X)
        dec = decomposition(m, X, X)
        inside = np.isclose(dec.membership, 1.0)
        self.assertGreater(inside.sum(), 0)
        spread = c.e[inside].std(axis=1)
        self.assertGreater((spread > 1e-12).mean(), 0.9)

    def test_does_not_change_predictions(self):
        m, X, _ = fitted(3)
        before = m.predict(X)
        contrastive(m, X); decomposition(m, X, X)
        np.testing.assert_array_equal(before, m.predict(X))


class TestContrastiveMulti(unittest.TestCase):
    def test_k1_ranks_like_contrastive(self):
        m, X, _ = fitted(5)
        a = contrastive_multi(m, X, k=1)
        c = contrastive(m, X)
        b = c.e
        nb = np.abs(b).sum(axis=1, keepdims=True)
        b = np.divide(b, nb, out=np.zeros_like(b), where=nb > 0)
        # Same competitor box (the two pick differently only on exact ties).
        Mo = np.where(m.box_class_[None] != c.cls_a[:, None], m.memberships(X), -np.inf)
        same = Mo.argmax(axis=1) == c.box_b
        self.assertGreater(same.mean(), 0.8)
        np.testing.assert_allclose(a[same], b[same], atol=1e-12)

    def test_deterministic_and_finite(self):
        m, X, _ = fitted(6)
        a, b = contrastive_multi(m, X), contrastive_multi(m, X)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(a).all())


class TestPaceFlip(unittest.TestCase):
    def test_flip_and_drop_equal_brute_force_rerun(self):
        m, X, _ = fitted(7)
        Xe, bg = X[:25], X[400:440]
        _, flip, drop = pace_flip(m, Xe, bg, n_quantiles=None, return_parts=True)
        det0 = m.predict_details(Xe)

        def margin(det, a):
            s = det.support_rescaled.copy()
            rows = np.arange(len(s))
            ma = s[rows, a].copy()
            s[rows, a] = -np.inf
            return ma - s.max(axis=1)

        for p in range(len(Xe)):
            a = det0.cls[p]
            m0 = margin(det0, det0.cls)[p]
            for i in range(Xe.shape[1]):
                Y = np.repeat(Xe[p][None], len(bg), axis=0)
                Y[:, i] = bg[:, i]
                det = m.predict_details(Y)
                self.assertEqual(flip[p, i], np.mean(det.cls != a))
                self.assertAlmostEqual(drop[p, i], m0 - margin(det, np.full(len(Y), a)).mean(),
                                       places=9)
        self.assertGreater(flip.sum(), 0)

    def test_deterministic_and_flip_order_kept(self):
        m, X, _ = fitted(8)
        a = pace_flip(m, X[:20], X[400:450])
        b = pace_flip(m, X[:20], X[400:450])
        np.testing.assert_array_equal(a, b)
        _, flip, _ = pace_flip(m, X[:20], X[400:450], return_parts=True)
        self.assertTrue(((a - flip) >= 0).all() and ((a - flip) <= 0.5 / 32 + 1e-12).all())


def problem(seed, n=300, d=5, classes=2, grid=None, label_noise=0.0):
    """Random data; ``grid`` rounds features to a coarse grid (many exact ties)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    if grid:
        X = np.round(X * grid) / grid
    score = X[:, 0] + 0.5 * X[:, min(1, d - 1)] + rng.normal(0, 0.1, n)
    y = np.digitize(score, np.quantile(score, np.linspace(0, 1, classes + 1)[1:-1]))
    if label_noise:
        noisy = rng.uniform(size=n) < label_noise
        y = np.where(noisy, rng.integers(classes, size=n), y)
    return X, y


class TestPaceFlipFast(unittest.TestCase):
    """The fast explainer must equal the original loop bit for bit."""

    def assertIdentical(self, m, X, bg, n_quantiles=32, explainer=None):
        ref = pace_flip_loop(m, X, bg, n_quantiles=n_quantiles, return_parts=True)
        ex = explainer or PaceFlip(m, bg, n_quantiles)
        new = ex.explain(X, return_parts=True)
        for name, u, v in zip(("score", "flip", "drop"), ref, new):
            np.testing.assert_array_equal(u, v, err_msg=name)
        return ex

    def test_identical_to_original_loop(self):
        rules = [("fixed", "search"), ("original", "best"), ("fixed", "best"), ("original", "search")]
        k = 0
        for d, classes in [(2, 2), (6, 3), (12, 4)]:
            for rule, expansion in rules:
                tol = (1e-9, 0.0)[k % 2]
                nq = (32, 8, None)[k % 3]
                with self.subTest(d=d, classes=classes, rule=rule, expansion=expansion,
                                  tie_tol=tol, n_quantiles=nq):
                    X, y = problem(k, d=d, classes=classes)
                    m = CFMNN(centroid_rule=rule, expansion=expansion, tie_tol=tol).fit(X[:250], y[:250])
                    self.assertIdentical(m, X[250:265], X[200:240], nq)
                k += 1

    def test_tie_break_path(self):
        # Duplicate points with different labels make boxes of different
        # classes tie at the top, so the nearest-centroid rule decides.
        for tol in (1e-9, 0.0):
            X, y = problem(11, n=400, d=4, classes=3, grid=2, label_noise=0.3)
            m = CFMNN(tie_tol=tol).fit(X[:300], y[:300])
            ex = self.assertIdentical(m, X[300:340], X[:60])
            self.assertGreater(ex.stats_["tie_break_cells"], 0)

    def test_single_class_model(self):
        X, _ = problem(12)
        m = CFMNN().fit(X[:200], np.zeros(200, dtype=int))
        ex = self.assertIdentical(m, X[200:220], X[:40])
        _, flip, _ = ex.explain(X[200:220], return_parts=True)
        self.assertFalse(flip.any())

    def test_chunks_reuse_and_rebuild(self):
        X, y = problem(13, d=6, classes=3)
        m = CFMNN().fit(X[:250], y[:250])
        Xe, bg = X[250:280], X[:50]
        for chunk in (1, 3 * m.n_boxes_ * 6, 10**9):   # one point, three points, all at once
            m.chunk_elems = chunk
            self.assertIdentical(m, Xe, bg)
        ex = PaceFlip(m, bg)
        np.testing.assert_array_equal(ex.explain(Xe), ex(Xe))
        # The model's boxes change (as in refinement): the tables must follow.
        m.W_[0] = np.minimum(m.W_[0] + 0.1, 1.0)
        self.assertIdentical(m, Xe, bg, explainer=ex)
        m.fit(X[30:280], y[30:280])                    # a different set of boxes
        self.assertIdentical(m, Xe, bg, explainer=ex)

    def test_large_background_in_blocks(self):
        X, y = problem(14, d=5)
        m = CFMNN().fit(X[:250], y[:250])
        ex = PaceFlip(m, X[:120], n_quantiles=None, table_elems=7 * m.n_boxes_ * 5)
        self.assertGreater(len(ex._blocks), 1)
        self.assertIdentical(m, X[250:270], X[:120], n_quantiles=None, explainer=ex)


class TestS3(unittest.TestCase):
    def test_rule_and_box_fit(self):
        ds = data.load("S3")
        np.testing.assert_array_equal(ds.y, data.s3_rule(ds.X))
        m = CFMNN().fit(ds.X[:4000], ds.y[:4000])
        self.assertGreater((m.predict(ds.X[4000:]) == ds.y[4000:]).mean(), 0.85)


class TestCounterfactual(unittest.TestCase):
    def test_valid_and_respects_locks_and_bounds(self):
        m, X, y = fitted(4)
        pred = m.predict(X[:60])
        locked = np.array([False, False, True, False, False])
        found = 0
        for x, p in zip(X[:60], pred):
            cf = counterfactual(m, x, 1 - p, locked=locked, X_train=X, y_train=y)
            if cf.found:
                found += 1
                self.assertEqual(m.predict(cf.x_cf[None])[0], 1 - p)
                self.assertEqual(cf.x_cf[2], x[2])
                self.assertTrue((cf.x_cf >= 0).all() and (cf.x_cf <= 1).all())
                self.assertGreaterEqual(cf.plausibility, 0)
        self.assertGreater(found, 40)


class TestGlobal(unittest.TestCase):
    def test_finds_relevant_features_on_synthetic(self):
        ds = data.make_synthetic("S1", n=3000, seed=0)
        m = CFMNN().fit(ds.X, ds.y)
        for g in (gi.data_based(m, ds.X[:1000]), gi.model_only(m)):
            self.assertEqual(set(np.argsort(-g)[:3]), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
