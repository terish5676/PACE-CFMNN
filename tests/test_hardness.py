"""Proposition 5: the voting-game construction behind the #P-hardness of exact PACE-SHAP."""

import unittest

import numpy as np

from pace_cfmnn.explain import PaceShap
from pace_cfmnn.hardness import shapley_shubik, shapley_shubik_dp, voting_game_model


def margin(model, X):
    det = model.predict_details(X)
    s = det.support_rescaled
    return s[:, 0] - s[:, 1]          # class A = 0 is fixed, as in the game


def random_game(rng, n):
    w = rng.integers(1, 12, size=n)
    q = int(rng.integers(1, w.sum() + 1))
    return w, q


class TestShapleyShubik(unittest.TestCase):
    def test_known_game(self):
        # weights (2, 1, 1), quota 3: player 0 is pivotal in 4 of the 6 orders.
        np.testing.assert_allclose(shapley_shubik([2, 1, 1], 3), [2 / 3, 1 / 6, 1 / 6])

    def test_dp_matches_enumeration(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            w, q = random_game(rng, int(rng.integers(1, 8)))
            np.testing.assert_allclose(shapley_shubik_dp(w, q), shapley_shubik(w, q),
                                       rtol=0, atol=1e-12)

    def test_indices_sum_to_one(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            w, q = random_game(rng, 6)
            self.assertAlmostEqual(shapley_shubik(w, q).sum(), 1.0, places=12)

    def test_rejects_bad_games(self):
        with self.assertRaises(ValueError):
            shapley_shubik([1.5, 2], 2)
        with self.assertRaises(ValueError):
            shapley_shubik([1, 2], 4)
        with self.assertRaises(ValueError):
            voting_game_model([1, 2], 2, gamma=1.0)


class TestConstruction(unittest.TestCase):
    def test_four_boxes_two_classes_in_unit_cube(self):
        g = voting_game_model([3, 1, 2], 4)
        m = g.model
        self.assertEqual(m.n_boxes_, 4)
        self.assertEqual(list(m.box_class_), [0, 0, 1, 1])
        for arr in (m.V_, m.W_, g.x, g.z):
            self.assertTrue(((arr >= 0) & (arr <= 1)).all())
        self.assertEqual(int(m.predict_details(g.x).cls[0]), 0)

    def test_margin_is_the_voting_game_for_every_coalition(self):
        rng = np.random.default_rng(2)
        for _ in range(8):
            n = int(rng.integers(2, 7))
            w, q = random_game(rng, n)
            g = voting_game_model(w, q)
            d = 2 * n
            masks = np.arange(1 << d)
            bits = ((masks[:, None] >> np.arange(d)[None, :]) & 1).astype(bool)
            H = np.where(bits, g.x, g.z)               # (x_S, z_rest) for every S
            wins = (bits[:, :n].astype(int) @ w) >= q
            np.testing.assert_allclose(margin(g.model, H), 2 * g.eps * wins,
                                       rtol=0, atol=1e-12)

    def test_pace_shap_is_scaled_shapley_shubik(self):
        rng = np.random.default_rng(3)
        for _ in range(12):
            n = int(rng.integers(1, 7))                # d = 2n <= 12: exact mode
            w, q = random_game(rng, n)
            g = voting_game_model(w, q)
            for prune in (True, False):
                ps = PaceShap(g.model, g.z, prune=prune)
                phi = ps.explain(g.x)[0]
                self.assertEqual(ps.stats_["mode"], "exact")
                np.testing.assert_allclose(phi[:n], 2 * g.eps * shapley_shubik(w, q),
                                           rtol=0, atol=1e-12)
                np.testing.assert_allclose(phi[n:], 0.0, rtol=0, atol=1e-12)

    def test_other_gamma(self):
        g = voting_game_model([5, 3, 3, 1], 7, gamma=8.0)
        phi = PaceShap(g.model, g.z).explain(g.x)[0]
        np.testing.assert_allclose(phi[:4], 2 * g.eps * shapley_shubik([5, 3, 3, 1], 7),
                                   rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
