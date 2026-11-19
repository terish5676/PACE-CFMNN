"""RQ1 extensions: PACE-SHAP-cond, KernelSHAP on a game, PACE-flip-cond,
interaction values, PACE-sparse, realism and global aggregation."""

import itertools
import math
import sys
import unittest
from pathlib import Path

import numpy as np

from pace_cfmnn import CFMNN, data
from pace_cfmnn import global_importance as gi
from pace_cfmnn import realism
from pace_cfmnn import xai_baselines as xb
from pace_cfmnn.counterfactual import counterfactual, sparse_counterfactual
from pace_cfmnn.explain import (PaceShap, _row_chunks, _row_weights, cond_neighbour_values,
                                cond_scale, pace_flip, pace_flip_cond)
from tests.test_explain import problem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def game_values(m, x, a, bg, sigma=None):
    """v(S) for every coalition (bitmask), by re-running the model on every hybrid."""
    d = len(x)
    scale = cond_scale(bg)
    v = np.zeros(1 << d)
    for mask in range(1 << d):
        S = [i for i in range(d) if mask >> i & 1]
        Z = bg.copy()
        Z[:, S] = x[S]
        f = xb.margin_score(m, Z, a)
        if sigma is None or not S:
            v[mask] = f.mean()
        else:
            d2 = (((x[S] - bg[:, S]) / scale[S]) ** 2).sum(axis=1) / len(S)
            w = np.exp(-d2 / (2 * sigma ** 2))
            v[mask] = (w * f).sum() / w.sum()
    return v


def shapley_from(v, d):
    phi = np.zeros(d)
    for i in range(d):
        for mask in range(1 << d):
            if mask >> i & 1:
                continue
            s = bin(mask).count("1")
            w = math.factorial(s) * math.factorial(d - s - 1) / math.factorial(d)
            phi[i] += w * (v[mask | 1 << i] - v[mask])
    return phi


def interaction_from(v, d, i, j):
    tot = 0.0
    for mask in range(1 << d):
        if mask >> i & 1 or mask >> j & 1:
            continue
        s = bin(mask).count("1")
        w = math.factorial(s) * math.factorial(d - s - 2) / math.factorial(d - 1)
        tot += w * (v[mask | 1 << i | 1 << j] - v[mask | 1 << i] - v[mask | 1 << j] + v[mask])
    return tot


def exact_values_before_sharing(ps, x, a):
    """PaceShap's v(S) table as computed before games shared their margins: one game
    (``ps.sigma``) per pass. Kept to show that sharing changed no number."""
    Delta, r_i, j_i, g_i, counts, has_other = ps._prepare(x, a)
    ind = ps._exact_tables[0]
    wt = None
    if ps.sigma is not None:
        wt = _row_weights(-(ps._q(x) @ ind) / (2.0 * ps.sigma ** 2 * np.maximum(ps._size, 1)))
    nm, R = ind.shape[1], len(counts)
    bounds = np.r_[0, np.cumsum(counts)]
    v = np.zeros(nm)
    for r0, r1 in _row_chunks(counts * nm, ps.chunk_elems):
        rr, jj = r_i[bounds[r0]:bounds[r1]], j_i[bounds[r0]:bounds[r1]]
        seg = 2 * (rr - r0) + g_i[bounds[r0]:bounds[r1]]
        vals = ps._MB[rr, jj][:, None] + Delta[rr, jj] @ ind
        cuts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
        gmax = np.maximum.reduceat(vals, cuts, axis=0)
        sid = seg[cuts]
        isa = sid % 2 == 0
        GA = np.full((r1 - r0, nm), -np.inf)
        GO = np.full((r1 - r0, nm), -np.inf)
        GA[sid[isa] // 2] = gmax[isa]
        GO[sid[~isa] // 2] = gmax[~isa]
        if wt is None:
            v += ps._margin(GA, GO, has_other).sum(axis=0)
        else:
            v += (wt[r0:r1] * ps._margin(GA, GO, has_other)).sum(axis=0)
    return v / R if wt is None else v


def kernel_shap_game_before_sharing(model, X, background, nsamples, seed, sigma=None):
    """``xb.kernel_shap_game`` as written before games shared their model calls."""
    X, B = model._check_X(X), model._check_X(background)
    n, d = X.shape
    R = len(B)
    scale = cond_scale(B)
    cls = model.predict_details(X).cls
    rng = np.random.default_rng(seed)
    out = np.zeros((n, d))
    for p in range(n):
        x, a = X[p], int(cls[p])
        Z, w = xb.shapley_kernel_coalitions(d, nsamples, rng)
        k = len(Z)
        hybrid = np.where(Z[:, None, :], x, B[None, :, :]).reshape(k * R, d)
        f = xb.margin_score(model, hybrid, a).reshape(k, R)
        if sigma is None:
            v = f.mean(axis=1)
        else:
            q = ((x - B) / scale) ** 2
            logw = -(Z.astype(float) @ q.T) / (2.0 * sigma ** 2 * Z.sum(axis=1)[:, None])
            wt = np.exp(logw - logw.max(axis=1, keepdims=True))
            v = (wt * f).sum(axis=1) / wt.sum(axis=1)
        f0 = xb.margin_score(model, B, a).mean()
        fx = xb.margin_score(model, x[None], a)[0]
        out[p] = xb.kernel_solve(Z, w, v - f0, fx - f0)
    return out


class TestPaceShapCond(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = problem(60, d=4, classes=3)
        cls.m = CFMNN().fit(X[:250], y[:250])
        cls.Xe, cls.bg = X[250:256], X[:15]
        cls.cls = cls.m.predict_details(cls.Xe).cls

    def test_exact_equals_brute_force_weighted_game(self):
        for sigma in (0.1, 0.5):
            phi = PaceShap(self.m, self.bg, sigma=sigma).explain(self.Xe)
            ref = [shapley_from(game_values(self.m, x, a, self.bg, sigma), 4)
                   for x, a in zip(self.Xe, self.cls)]
            np.testing.assert_allclose(phi, ref, atol=1e-12)

    def test_large_sigma_is_pace_shap(self):
        for exact_max_d in (12, 0):
            a = PaceShap(self.m, self.bg, sigma=1e8, exact_max_d=exact_max_d, seed=2).explain(self.Xe)
            if exact_max_d:
                b = PaceShap(self.m, self.bg).explain(self.Xe)
                np.testing.assert_allclose(a, b, atol=1e-9)
            else:
                # Sampled: same orders for every row, so compare with the exact values loosely.
                b = PaceShap(self.m, self.bg).explain(self.Xe)
                self.assertLess(np.abs(a - b).max(), 0.25 * np.abs(b).max())

    def test_efficiency_both_modes(self):
        margin = np.array([xb.margin_score(self.m, x[None], a)[0] for x, a in zip(self.Xe, self.cls)])
        base = np.array([xb.margin_score(self.m, self.bg, a).mean() for a in self.cls])
        for exact_max_d in (12, 0):
            phi, fx, f0 = PaceShap(self.m, self.bg, sigma=0.2, exact_max_d=exact_max_d).explain(
                self.Xe, return_parts=True)
            np.testing.assert_allclose(fx, margin, atol=1e-12)
            np.testing.assert_allclose(f0, base, atol=1e-12)
            np.testing.assert_allclose(phi.sum(1), fx - f0, atol=1e-12)

    def test_sampled_converges_and_ignores_chunks(self):
        exact = PaceShap(self.m, self.bg, sigma=0.3).explain(self.Xe)
        a = PaceShap(self.m, self.bg, sigma=0.3, exact_max_d=0, n_permutations=2000, seed=1).explain(self.Xe)
        np.testing.assert_allclose(a, exact, atol=0.05 * np.abs(exact).max())
        b = PaceShap(self.m, self.bg, sigma=0.3, exact_max_d=0, n_permutations=2000, seed=1,
                     chunk_elems=50).explain(self.Xe)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_two_features_antithetic_is_exact(self):
        X, y = problem(61, d=2, classes=2)
        m = CFMNN().fit(X[:200], y[:200])
        Xe, bg = X[200:210], X[:20]
        np.testing.assert_allclose(
            PaceShap(m, bg, sigma=0.2, exact_max_d=0, n_permutations=2).explain(Xe),
            PaceShap(m, bg, sigma=0.2).explain(Xe), atol=1e-12)

    def test_rejects_bad_sigma(self):
        with self.assertRaises(ValueError):
            PaceShap(self.m, self.bg, sigma=0.0)

    def test_games_at_once_equal_separate_and_before_sharing(self):
        sigmas = (None, 0.1, 0.5)
        for chunk in (4_000_000, 50):                    # one chunk per point, and many
            times = {}
            got = PaceShap(self.m, self.bg, chunk_elems=chunk).explain_games(self.Xe, sigmas,
                                                                             times=times)
            self.assertEqual(len(times["own"]), 3)
            for g, s in zip(got, sigmas):
                ps = PaceShap(self.m, self.bg, sigma=s, chunk_elems=chunk)
                np.testing.assert_array_equal(g, ps.explain(self.Xe))
                for x, a in zip(self.Xe, self.cls):
                    np.testing.assert_array_equal(ps._v_exact(x, int(a))[0],
                                                  exact_values_before_sharing(ps, x, int(a)))
        with self.assertRaises(ValueError):
            PaceShap(self.m, self.bg, exact_max_d=0).explain_games(self.Xe, (None,))
        with self.assertRaises(ValueError):
            PaceShap(self.m, self.bg).explain_games(self.Xe, (0.0,))


class TestKernelShapGame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = problem(70, d=4, classes=3)
        cls.m = CFMNN().fit(X[:250], y[:250])
        cls.Xe, cls.bg = X[250:256], X[:12]

    def test_every_coalition_equals_pace_shap(self):
        for sigma in (None, 0.3):
            ks = xb.kernel_shap_game(self.m, self.Xe, self.bg, nsamples=14, sigma=sigma)
            ps = PaceShap(self.m, self.bg, sigma=sigma).explain(self.Xe)
            np.testing.assert_allclose(ks, ps, atol=1e-10)

    @unittest.skipUnless(__import__("importlib").util.find_spec("shap"), "shap not installed")
    def test_every_coalition_equals_shap_library(self):
        ks = xb.kernel_shap_game(self.m, self.Xe, self.bg, nsamples=14)
        ref = xb.kernel_shap(self.m, self.Xe, self.bg, nsamples=2048, target="margin", l1_reg=False)
        np.testing.assert_allclose(ks, ref, atol=1e-10)

    def test_sampled_is_efficient_and_seeded(self):
        X, y = problem(71, d=8, classes=3)
        m = CFMNN().fit(X[:250], y[:250])
        Xe, bg = X[250:254], X[:20]
        cls = m.predict_details(Xe).cls
        for sigma in (None, 0.3):
            a = xb.kernel_shap_game(m, Xe, bg, nsamples=40, seed=3, sigma=sigma)
            fx = np.array([xb.margin_score(m, x[None], c)[0] for x, c in zip(Xe, cls)])
            f0 = np.array([xb.margin_score(m, bg, c).mean() for c in cls])
            np.testing.assert_allclose(a.sum(1), fx - f0, atol=1e-10)
            np.testing.assert_array_equal(a, xb.kernel_shap_game(m, Xe, bg, nsamples=40, seed=3,
                                                                 sigma=sigma))

    def test_games_at_once_equal_separate_and_before_sharing(self):
        X, y = problem(71, d=8, classes=3)
        m = CFMNN().fit(X[:250], y[:250])
        Xe, bg = X[250:254], X[:20]
        sigmas = (None, 0.3, 0.1)
        times = {}
        got = xb.kernel_shap_games(m, Xe, bg, nsamples=40, seed=3, sigmas=sigmas, times=times)
        self.assertEqual(len(times["own"]), 3)
        for g, s in zip(got, sigmas):
            ref = kernel_shap_game_before_sharing(m, Xe, bg, 40, 3, sigma=s)
            np.testing.assert_array_equal(g, ref)
            np.testing.assert_array_equal(xb.kernel_shap_game(m, Xe, bg, nsamples=40, seed=3,
                                                              sigma=s), ref)

    def test_coalitions(self):
        rng = np.random.default_rng(0)
        Z, w = xb.shapley_kernel_coalitions(6, 40, rng)
        self.assertEqual(Z.shape, (40, 6))
        np.testing.assert_array_equal(Z[1::2], ~Z[0::2])          # complementary pairs
        s = Z.sum(1)
        self.assertTrue(((s >= 1) & (s <= 5)).all())
        np.testing.assert_array_equal(w, 1.0)
        Z, w = xb.shapley_kernel_coalitions(4, 100, rng)            # every coalition
        self.assertEqual(len(Z), 14)
        self.assertEqual(len({tuple(z) for z in Z}), 14)


class TestPaceFlipCond(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = problem(80, d=5, classes=3)
        cls.m = CFMNN().fit(X[:250], y[:250])
        cls.Xe, cls.bg = X[250:262], X[:60]

    def test_neighbour_values_naive(self):
        scale = cond_scale(self.bg)
        x = self.Xe[0]
        vals = cond_neighbour_values(x, self.bg, k=7)
        for i in range(5):
            other = [k for k in range(5) if k != i]
            dist = (((x[other] - self.bg[:, other]) / scale[other]) ** 2).sum(1)
            near = np.argsort(dist, kind="stable")[:7]
            np.testing.assert_array_equal(np.sort(vals[:, i]), np.sort(self.bg[near, i]))

    def test_equals_occlusion_with_same_values(self):
        vals = np.stack([cond_neighbour_values(x, self.bg, k=16) for x in self.Xe])
        np.testing.assert_allclose(pace_flip_cond(self.m, self.Xe, self.bg, k=16),
                                   xb.occlusion(self.m, self.Xe, None, values=vals), atol=1e-9)

    def test_all_rows_equals_pace_flip_with_all_rows(self):
        np.testing.assert_allclose(pace_flip_cond(self.m, self.Xe, self.bg, k=len(self.bg)),
                                   pace_flip(self.m, self.Xe, self.bg, n_quantiles=None), atol=1e-9)


class TestInteractions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = problem(90, d=4, classes=3)
        cls.m = CFMNN().fit(X[:250], y[:250])
        cls.Xe, cls.bg = X[250:255], X[:10]
        cls.cls = cls.m.predict_details(cls.Xe).cls

    def test_equals_brute_force(self):
        for sigma in (None, 0.3):
            inter = PaceShap(self.m, self.bg, sigma=sigma).interactions(self.Xe)
            for p, (x, a) in enumerate(zip(self.Xe, self.cls)):
                v = game_values(self.m, x, a, self.bg, sigma)
                for i, j in itertools.combinations(range(4), 2):
                    self.assertAlmostEqual(inter[p, i, j], interaction_from(v, 4, i, j) / 2, places=12)

    def test_symmetric_and_rows_sum_to_shapley(self):
        for sigma in (None, 0.3):
            ex = PaceShap(self.m, self.bg, sigma=sigma)
            inter = ex.interactions(self.Xe)
            np.testing.assert_allclose(inter, inter.transpose(0, 2, 1), atol=0)
            np.testing.assert_allclose(inter.sum(axis=2), ex.explain(self.Xe), atol=1e-12)

    def test_sampled_mode_refuses(self):
        with self.assertRaises(ValueError):
            PaceShap(self.m, self.bg, exact_max_d=2).interactions(self.Xe)


class TestPaceSparse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = problem(100, n=400, d=4, classes=3)
        cls.m = CFMNN().fit(X[:300], y[:300])
        cls.X = X[300:340]
        det = cls.m.predict_details(cls.X)
        cls.targets = [cls.m.classes_[b] for b in det.runner_up]

    def brute_min(self, x, target, steps=(0.25, 0.5, 0.75, 1.0)):
        """Fewest changed features over the same moves, checked by calling the model."""
        m = self.m
        t = int(np.flatnonzero(m.classes_ == target)[0])
        best = None
        for j in m.class_boxes_[t]:
            clipped = np.clip(x, m.V_[j], m.W_[j])
            for s in (0.0,) + steps:
                vals = np.clip(clipped + s * (m.centroid_[j] - clipped), m.V_[j], m.W_[j])
                F = np.flatnonzero(np.abs(vals - x) > 1e-12)
                for size in range(1, len(F) + 1):
                    if best is not None and size >= best:
                        break
                    for S in itertools.combinations(F, size):
                        c = x.copy()
                        c[list(S)] = vals[list(S)]
                        if m.predict(c[None])[0] == target:
                            best = size
                            break
        return best

    def test_equals_brute_force_minimum(self):
        n_boxes = max(len(b) for b in self.m.class_boxes_)
        for x, target in zip(self.X[:15], self.targets[:15]):
            cf = sparse_counterfactual(self.m, x, target, max_boxes=n_boxes)
            want = self.brute_min(x, target)
            self.assertTrue(cf.found)
            self.assertEqual(int(np.sum(np.abs(cf.x_cf - x) > 1e-9)), want)

    def test_valid_and_never_more_than_pace(self):
        # A budget of 1 subset forces the greedy search, except where a box needs all of its
        # outside features moved (a single subset, still searched exactly).
        for max_subsets in (65_536, 1):
            hows = set()
            for x, target in zip(self.X, self.targets):
                cf = sparse_counterfactual(self.m, x, target, max_subsets=max_subsets)
                pace = counterfactual(self.m, x, target)
                if not cf.found:
                    self.assertFalse(pace.found)
                    continue
                self.assertEqual(self.m.predict(cf.x_cf[None])[0], target)
                if pace.found:
                    self.assertLessEqual(np.sum(np.abs(cf.x_cf - x) > 1e-9),
                                         np.sum(np.abs(pace.x_cf - x) > 1e-9))
                hows.add(cf.how)
            if max_subsets == 1:
                self.assertIn("greedy", hows)

    def test_locked_features_kept(self):
        lock = np.array([True, False, False, False])
        for x, target in zip(self.X, self.targets):
            cf = sparse_counterfactual(self.m, x, target, locked=lock)
            if cf.found:
                self.assertEqual(cf.x_cf[0], x[0])
                self.assertEqual(self.m.predict(cf.x_cf[None])[0], target)


class TestRealism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        cls.Xtr = rng.uniform(size=(500, 3))
        cls.man = realism.Manifold(cls.Xtr)

    def test_manifold_rule(self):
        self.assertEqual(self.man.share(self.Xtr), 0.0)             # training rows: distance 0
        self.assertEqual(self.man.share(self.Xtr + 5.0), 1.0)        # far away: all off
        new = np.random.default_rng(6).uniform(size=(2000, 3))
        self.assertLess(self.man.share(new), 0.2)

    def test_shapley_inputs_are_hybrids(self):
        rng = np.random.default_rng(1)
        X, B = self.Xtr[:4], self.Xtr[100:130]
        for sigma in (None, 0.2):
            Z = realism.shapley_inputs(X, B, 50, rng, sigma=sigma)
            self.assertEqual(Z.shape, (4, 50, 3))
            for p in range(4):
                for z in Z[p]:
                    from_x = z == X[p]
                    rest = ~from_x
                    if rest.any():
                        self.assertTrue((B[:, rest] == z[rest]).all(axis=1).any())

    def test_conditional_rows_are_nearest_for_tiny_sigma(self):
        rng = np.random.default_rng(2)
        X, B = self.Xtr[:3], self.Xtr[100:160]
        scale = cond_scale(B)
        Z = realism.shapley_inputs(X, B, 40, rng, sigma=1e-6)
        for p in range(3):
            for z in Z[p]:
                S = z == X[p]
                if S.all() or not S.any():
                    continue
                d2 = (((X[p][S] - B[:, S]) / scale[S]) ** 2).sum(1)
                self.assertTrue(np.allclose(z[~S], B[np.argmin(d2), ~S]))

    def test_occlusion_and_lime_inputs(self):
        rng = np.random.default_rng(3)
        X = self.Xtr[:5]
        vals = self.Xtr[200:208]
        Z = realism.occlusion_inputs(X, vals, 30, rng)
        self.assertTrue(((Z != X[:, None, :]).sum(axis=2) <= 1).all())
        Z, w = realism.lime_inputs(X, self.Xtr, 30, rng)
        self.assertEqual(Z.shape, (5, 30, 3))
        self.assertTrue(((w > 0) & (w <= 1)).all())

    def test_conditional_curve_rows_are_near(self):
        rng = np.random.default_rng(4)
        X, B = self.Xtr[:3], self.Xtr[300:340]
        scale = cond_scale(B)
        Z = realism.curve_inputs(X, B, 40, rng, conditional=True, reps=5)
        for p in range(3):
            for z in Z[p]:
                S = z == X[p]
                if S.all() or not S.any():
                    continue
                d2 = (((X[p][S] - B[:, S]) / scale[S]) ** 2).sum(1)
                near = np.argsort(d2, kind="stable")[:5]
                self.assertTrue(any(np.allclose(z[~S], B[r, ~S]) for r in near))


class TestConditionalCurves(unittest.TestCase):
    def test_single_repeated_row_equals_interventional_test(self):
        import run_faithfulness as rf
        X, y = problem(110, d=5, classes=3)
        m = CFMNN().fit(X[:250], y[:250])
        Xe = X[250:270]
        attr = np.random.default_rng(0).normal(size=Xe.shape)
        bg = np.repeat(X[:1], 6, axis=0)          # every row is the same: no choice to make
        np.testing.assert_allclose(rf.cond_curves(m, Xe, attr, bg)[1],
                                   rf.deletion_auc(m, Xe, attr, bg, signed=True)[1], atol=1e-12)
        np.testing.assert_allclose(rf.cond_curves(m, Xe, attr, bg, insertion=True)[1],
                                   rf.insertion_auc(m, Xe, attr, bg)[1], atol=1e-12)

    def test_rows_equal_to_the_point_keep_everything(self):
        import run_faithfulness as rf
        X, y = problem(111, d=4, classes=2)
        m = CFMNN().fit(X[:250], y[:250])
        x = X[250:251]
        attr = np.ones((1, 4))
        kept, marg = rf.cond_curves(m, x, attr, np.repeat(x, 5, axis=0))
        np.testing.assert_allclose([kept[0], marg[0]], [1.0, 1.0])


class TestCondMethods(unittest.TestCase):
    """The cond protocol's explainers: the pairs share their model calls, results unchanged."""

    def check(self, d):
        import run_faithfulness as rf
        X, y = problem(120 + d, d=d, classes=3)
        m = CFMNN().fit(X[:250], y[:250])
        Xe, B = X[250:253], X[:40]
        got = {}
        for group in rf.methods_cond(m, Xe, B):
            got.update(group(4))
        self.assertEqual(list(got), ["PACE-SHAP", "PACE-SHAP-cond", "PACE-flip", "PACE-flip-cond",
                                     "KernelSHAP", "KernelSHAP-cond", "LIME", "Random"])
        apart = {
            "PACE-SHAP": PaceShap(m, B, n_permutations=rf.SHAP_V3_P, seed=4).explain(Xe),
            "PACE-SHAP-cond": PaceShap(m, B, n_permutations=rf.SHAP_V3_P, seed=4,
                                       sigma=rf.COND_SIGMA).explain(Xe),
            "KernelSHAP": kernel_shap_game_before_sharing(m, Xe, B, rf.COND_NSAMPLES, 4),
            "KernelSHAP-cond": kernel_shap_game_before_sharing(m, Xe, B, rf.COND_NSAMPLES, 4,
                                                               sigma=rf.COND_SIGMA),
        }
        for name, ref in apart.items():
            np.testing.assert_array_equal(got[name][0], ref)
        for attr, sec in got.values():
            self.assertEqual(attr.shape, Xe.shape)
            self.assertGreaterEqual(sec, 0.0)

    def test_exact_mode(self):
        self.check(5)

    def test_sampled_mode(self):
        self.check(13)


class TestGlobalAndData(unittest.TestCase):
    def test_normalised_mean(self):
        s = np.array([[1.0, 1.0], [0.0, 2.0], [0.0, 0.0]])
        np.testing.assert_allclose(gi.normalised_mean(s), [1 / 6, 1 / 2])

    def test_correlated_synthetic_names(self):
        ds = data.load("S1c", seed=0)
        self.assertEqual(ds.name, "S1c")
        self.assertTrue(ds.info["correlated"])
        self.assertGreater(np.corrcoef(ds.X[:, 0], ds.X[:, 3])[0, 1], 0.9)


if __name__ == "__main__":
    unittest.main()
