"""Tests for the trust signals (blueprint 6, 11.2: neighborhood, signals, pace_score)."""

import inspect
import unittest

import numpy as np

from pace_cfmnn import CFMNN
from pace_cfmnn.evaluate import error_detection_metrics
from pace_cfmnn.explain import contrastive, contrastive_pair
from pace_cfmnn.neighborhood import Perturber
from pace_cfmnn.pace_score import PRS
from pace_cfmnn.signals import SignalComputer, Signals, _ei_terms, uncertainty


def fitted(seed=0, n=800, d=5, binary=False):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    if binary:
        X[:, -1] = (rng.uniform(size=n) < 0.3).astype(float)
    y = ((X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.15, n)) > 0.75).astype(int)
    return CFMNN().fit(X, y), X, y


class TestPerturber(unittest.TestCase):
    def test_shape_range_and_reproducible(self):
        _, X, _ = fitted(binary=True)
        p = Perturber(kappa=0.1, k=7).fit(X)
        Z = p.sample(X[:30], seed=3)
        self.assertEqual(Z.shape, (30, 7, X.shape[1]))
        self.assertTrue(((Z >= 0) & (Z <= 1)).all())
        np.testing.assert_array_equal(Z, p.sample(X[:30], seed=3))
        self.assertFalse(np.array_equal(Z, p.sample(X[:30], seed=4)))

    def test_discrete_features_keep_training_values(self):
        _, X, _ = fitted(binary=True)
        p = Perturber(k=50, p_discrete=0.5).fit(X)
        self.assertTrue(p.discrete_[-1])
        self.assertFalse(p.discrete_[:-1].any())
        Z = p.sample(X[:40], seed=0)
        self.assertTrue(np.isin(Z[:, :, -1], [0.0, 1.0]).all())
        self.assertTrue((Z[:, :, -1] != X[:40, None, -1]).any())  # some redraws change the value

    def test_noise_scales_with_training_std(self):
        _, X, _ = fitted()
        p = Perturber(kappa=0.05, k=400).fit(X)
        x = np.full((1, X.shape[1]), 0.5)  # far from the clip edges
        Z = p.sample(x, seed=0)[0]
        np.testing.assert_allclose(Z.std(axis=0), 0.05 * X.std(axis=0), rtol=0.15)

    def test_zero_noise_returns_copies(self):
        _, X, _ = fitted(binary=True)
        p = Perturber(kappa=0.0, k=5, p_discrete=0.0).fit(X)
        Z = p.sample(X[:10], seed=0)
        np.testing.assert_array_equal(Z, np.repeat(X[:10, None, :], 5, axis=1))


class TestContrastivePair(unittest.TestCase):
    def test_matches_contrastive_for_predicted_pair(self):
        m, X, _ = fitted(1)
        det = m.predict_details(X)
        a, b = contrastive(m, X), contrastive_pair(m, X, det.cls, det.runner_up)
        np.testing.assert_array_equal(a.e, b.e)
        np.testing.assert_array_equal(a.mode, b.mode)
        np.testing.assert_array_equal(a.box_a, b.box_a)
        np.testing.assert_array_equal(a.box_b, b.box_b)

    def test_pair_is_kept_when_prediction_changes(self):
        m, X, _ = fitted(2)
        det = m.predict_details(X)
        flipped = contrastive_pair(m, X, det.runner_up.clip(0), det.cls)  # "why B, not A"
        self.assertTrue((flipped.cls_a == det.runner_up.clip(0)).all())
        ok = flipped.mode == "membership"
        self.assertTrue((flipped.gap[ok] <= 1e-9).all())  # B is not ahead of A at x


class TestEITerms(unittest.TestCase):
    def test_identical_opposite_and_mode_switch(self):
        e = np.array([[0.3, -0.1, 0.05, 0.0]])
        same = _ei_terms(e, e, np.array(["membership"]), np.array(["membership"]), 0.5, 2)
        self.assertAlmostEqual(same[0], 0.0)
        opp = _ei_terms(e, -e, np.array(["membership"]), np.array(["membership"]), 0.5, 2)
        self.assertAlmostEqual(opp[0], 0.5)  # cosine term only: same top features
        sw = _ei_terms(e, e, np.array(["membership"]), np.array(["tie"]), 0.5, 2)
        self.assertEqual(sw[0], 1.0)


class TestSignals(unittest.TestCase):
    def test_no_noise_gives_zero_pd_and_ei(self):
        m, X, _ = fitted(3)
        p = Perturber(kappa=0.0, k=5, p_discrete=0.0).fit(X)
        sig = SignalComputer(m, p).compute(X[:100], seed=0)
        self.assertTrue((sig.pd == 0).all())
        self.assertTrue((sig.box_switch == 0).all())
        self.assertTrue((sig.mode_switch == 0).all())
        ok = np.isfinite(sig.ei)
        self.assertGreater(ok.mean(), 0.9)
        np.testing.assert_allclose(sig.ei[ok], 0.0, atol=1e-12)

    def test_pd_matches_a_direct_rerun(self):
        m, X, _ = fitted(4)
        p = Perturber(kappa=0.1, k=20).fit(X)
        pts = X[:25]
        sig = SignalComputer(m, p).compute(pts, seed=9)
        Z = p.sample(pts, np.random.default_rng(9))  # same draws, one chunk
        cls0 = m.predict_details(pts).cls
        clsZ = m.predict_details(Z.reshape(-1, X.shape[1])).cls.reshape(25, 20)
        np.testing.assert_allclose(sig.pd, (clsZ != cls0[:, None]).mean(axis=1))

    def test_ranges_and_reproducible(self):
        m, X, _ = fitted(5, binary=True)
        p = Perturber(kappa=0.05, k=10).fit(X)
        a = SignalComputer(m, p).compute(X[:60], seed=1)
        b = SignalComputer(m, p).compute(X[:60], seed=1)
        np.testing.assert_array_equal(a.pd, b.pd)
        np.testing.assert_array_equal(a.ei, b.ei)
        for v in (a.pd, a.u, a.u_star, a.box_switch, a.mode_switch, a.margin_unc, a.entropy):
            self.assertTrue(((v >= 0) & (v <= 1)).all())
        ei = a.ei[np.isfinite(a.ei)]
        self.assertTrue(((ei >= 0) & (ei <= 1)).all())

    def test_no_labels_anywhere(self):
        for fn in (SignalComputer.__init__, SignalComputer.compute, Perturber.fit,
                   Perturber.sample, uncertainty, PRS.fit, PRS.score):
            self.assertFalse({"y", "y_true", "labels"} & set(inspect.signature(fn).parameters))


class TestUncertainty(unittest.TestCase):
    def test_ties_are_graded(self):
        m, X, _ = fitted(6)
        det = m.predict_details(X)
        u, u_star, plain, _, tie = uncertainty(m, det)
        rows = np.arange(len(X))
        s = det.support_rescaled
        same = (det.runner_up >= 0) & np.isclose(s[rows, det.cls], s[rows, det.runner_up.clip(0)])
        np.testing.assert_array_equal(tie, same)
        self.assertGreater(tie.sum(), 5)
        self.assertTrue(np.allclose(plain[tie], 1.0))       # plain margin cannot tell ties apart
        self.assertGreater(np.ptp(u[tie]), 0.05)             # U can
        self.assertGreater(np.ptp(u_star[tie]), 0.02)        # so can U*
        self.assertTrue(((u >= 0) & (u <= 1)).all())
        self.assertTrue(((u_star >= 0) & (u_star <= 1)).all())

    def test_u_star_ranks_every_tie_first(self):
        m, X, _ = fitted(7)
        _, u_star, plain, _, tie = uncertainty(m, m.predict_details(X))
        self.assertGreater(u_star[tie].min(), u_star[~tie].max())
        # Outside ties U* keeps the plain margin's order.
        o = np.argsort(plain[~tie], kind="stable")
        self.assertTrue((np.diff(u_star[~tie][o]) >= -1e-12).all())


class TestPRS(unittest.TestCase):
    def _sig(self, pd, ei, u):
        z = np.zeros(len(pd))
        return Signals(np.asarray(pd, float), np.asarray(ei, float), np.asarray(u, float),
                       z, z, z, z)

    def test_ranks_come_from_inner_data_only(self):
        inner = self._sig([0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8])
        prs = PRS().fit(inner)
        np.testing.assert_allclose(prs.rank("pd", [0.15, 1.0, -1.0]), [0.5, 1.0, 0.0])
        a, _ = prs.score(self._sig([0.15], [0.25], [0.65]))
        b, _ = prs.score(self._sig([0.15, 0.9, 0.0], [0.25, 0.9, 0.0], [0.65, 0.9, 0.0]))
        self.assertEqual(a[0], b[0])  # other scored points do not move a point's score
        self.assertAlmostEqual(a[0], 0.5)

    def test_missing_ei_uses_pd_u_fallback(self):
        prs = PRS().fit(self._sig([0, 0.5, 1.0], [0, 0.5, 1.0], [0, 0.5, 1.0]))
        s, fb = prs.score(self._sig([1.0, 1.0], [np.nan, 0.0], [0.0, 0.0]))
        self.assertEqual(list(fb), [True, False])
        self.assertAlmostEqual(s[0], (1.0 + 1 / 3) / 2)
        self.assertAlmostEqual(s[1], (1.0 + 1 / 3 + 1 / 3) / 3)


class TestTrustBaselines(unittest.TestCase):
    def test_trust_score_rises_towards_the_other_class(self):
        from pace_cfmnn.trust_baselines import trust_score
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0.25, 0.05, (200, 2)), rng.normal(0.75, 0.05, (200, 2))])
        y = np.repeat([0, 1], 200)
        pts = np.array([[0.25, 0.25], [0.5, 0.5], [0.7, 0.7]])
        s = trust_score(X, y, pts, np.array([0, 0, 0]))
        self.assertTrue(s[0] < s[1] < s[2])           # less trust as x moves towards class 1
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_order_ensemble_and_knn(self):
        from pace_cfmnn.trust_baselines import knn_disagreement, order_ensemble_disagreement
        m, X, y = fitted(8)
        pred = m.predict(X[:100])
        dis = order_ensemble_disagreement(m, X, y, X[:100], pred, n_models=3, seed=0)
        self.assertTrue(((dis >= 0) & (dis <= 1)).all())
        self.assertGreater(dis.max(), 0)              # training order matters for CFMNN
        k = knn_disagreement(X, y, X[:100], pred, k=10)
        self.assertTrue(((k >= 0) & (k <= 1)).all())
        np.testing.assert_allclose(k * 10, np.round(k * 10))

    def test_same_order_ensemble_agrees(self):
        from sklearn.base import clone
        m, X, y = fitted(9)
        twin = clone(m).fit(X, y)                     # same rows, same order
        np.testing.assert_array_equal(twin.predict(X), m.predict(X))


class TestErrorDetectionMetrics(unittest.TestCase):
    def test_perfect_and_tied_scores(self):
        err = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
        perfect = error_detection_metrics(err, err.astype(float), budget=0.2)
        self.assertEqual((perfect["auroc"], perfect["auprc"]), (1.0, 1.0))
        self.assertEqual((perfect["precision_at"], perfect["recall_at"]), (1.0, 1.0))
        flat = error_detection_metrics(err, np.zeros(10), budget=0.2)
        self.assertAlmostEqual(flat["auroc"], 0.5)
        self.assertAlmostEqual(flat["precision_at"], 0.2)  # ties shared out: the base rate


if __name__ == "__main__":
    unittest.main()


class TestTrustCombine(unittest.TestCase):
    def test_box_evidence_is_zero_on_own_members(self):
        from pace_cfmnn.signals import box_evidence
        m, X, _ = fitted(10)
        det = m.predict_details(X)
        ev = box_evidence(m, X, X, det)
        self.assertTrue((ev >= 0).all())
        own = np.array([r in set(m.members_[det.winner[r]]) for r in range(len(X))])
        self.assertGreater(own.sum(), 100)
        np.testing.assert_allclose(ev[own], 0.0, atol=1e-6)   # sklearn distance rounding

    def test_order_members_match_the_baseline(self):
        from pace_cfmnn.trust_baselines import order_ensemble_disagreement
        from pace_cfmnn.trust_combine import order_members, trust_features
        m, X, y = fitted(11)
        members = order_members(m, X, y, n_models=3, seed=4)
        F, pred = trust_features(m, X, y, X[:100], members)
        np.testing.assert_array_equal(
            F["Order ensemble"], order_ensemble_disagreement(m, X, y, X[:100], pred, 3, seed=4))
        self.assertEqual(set(F), {"U*", "Margin", "Tie", "Box evidence", "Box count",
                                  "Order ensemble", "kNN", "Trust Score"})

    def test_learned_combiner(self):
        from pace_cfmnn.trust_combine import LearnedCombiner
        rng = np.random.default_rng(0)
        err = rng.uniform(size=400) < 0.2
        F = {"good": err + rng.normal(0, 0.3, 400), "noise": rng.normal(size=400)}
        c = LearnedCombiner(("good", "noise")).fit(F, err)
        m = error_detection_metrics(err, c.score(F))
        self.assertGreater(m["auprc"], 0.8)
        flat = LearnedCombiner(("good",)).fit(F, np.zeros(400, bool))
        self.assertEqual(len(set(flat.score(F))), 1)          # nothing to learn: constant
