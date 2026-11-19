"""Tests for scripts/run_gfmm_exact.py (Run P4): the three ways agree on a toy GFMM model, the check
fails as it should, and one small end-to-end task passes."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

HAS_HB = importlib.util.find_spec("hbbrain") is not None


@unittest.skipUnless(HAS_HB, "hyperbox-brain not installed")
class TestRun(unittest.TestCase):
    def test_exact_explainer_matches_on_a_toy_model(self):
        import run_gfmm_exact as ge
        from pace_cfmnn.gfmm_explain import GFMMBoxes, PaceShapMin, brute_force_min
        from tests.test_gfmm_explain import fit_gfmm
        est, X = fit_gfmm(3, 5, "iol")
        bx = GFMMBoxes.from_hbbrain(est)
        B, pts = X[:15], X[20:24]
        phi = PaceShapMin(bx, B).explain(pts)
        self.assertLessEqual(np.abs(phi - ge.exact_shap_min(bx, pts, B)).max(), ge.TOL_EXACT)
        ref = np.array([brute_force_min(bx, x, B, chunk_elems=500) for x in pts])   # many small blocks
        self.assertLessEqual(np.abs(phi - ref).max(), ge.TOL_BRUTE)

    def test_exact_explainer_arm_works_in_blocks(self):
        """P4 Amendment 2: the ExactExplainer arm never builds more than CHUNK_ELEMS
        (row, box, feature) numbers at once, and its values are unchanged."""
        import run_gfmm_exact as ge
        from pace_cfmnn.gfmm_explain import GFMMBoxes, PaceShapMin
        from tests.test_gfmm_explain import fit_gfmm
        est, X = fit_gfmm(3, 5, "iol")
        bx = GFMMBoxes.from_hbbrain(est)
        B, pts = X[:15], X[20:24]
        Z = X[:50]
        np.testing.assert_array_equal(ge.blocked_margin(bx, Z, 1, chunk_elems=300),
                                      bx.margin(Z, np.full(len(Z), 1)))
        sizes, margin = [], bx.margin
        bx.margin = lambda Zb, c: (sizes.append(len(Zb) * len(bx.V) * Zb.shape[1]), margin(Zb, c))[1]
        old = ge.CHUNK_ELEMS
        ge.CHUNK_ELEMS = 4 * len(bx.V) * 5
        try:
            got = ge.exact_shap_min(bx, pts, B)
        finally:
            ge.CHUNK_ELEMS = old
        self.assertLessEqual(max(sizes), 4 * len(bx.V) * 5)
        self.assertLessEqual(np.abs(got - PaceShapMin(bx, B).explain(pts)).max(), ge.TOL_EXACT)

    def test_check_fails_as_it_should(self):
        import run_gfmm_exact as ge
        row = {"part": "check", "dataset": "S1", "seed": 1, "model": "IOL-GFMM", "adapter_predicts_alike": True,
               "arm": ge.PACE, "max_abs_diff_brute": 0.0, "max_abs_diff_pace": 0.0}
        self.assertTrue(ge.check(pd.DataFrame([row]))[0])
        self.assertFalse(ge.check(pd.DataFrame([{**row, "max_abs_diff_brute": 1e-10}]))[0])
        self.assertFalse(ge.check(pd.DataFrame([{**row, "adapter_predicts_alike": False}]))[0])
        self.assertFalse(ge.check(pd.DataFrame([{**row, "arm": ge.EXACT, "max_abs_diff_pace": 1e-8}]))[0])

    def test_one_small_task(self):
        import run_gfmm_exact as ge
        old = dict(ge.N_POINTS)
        ge.N_POINTS["check"] = 1
        try:
            df = pd.DataFrame(ge.run_one("shuttle", 194, "check"))
        finally:
            ge.N_POINTS.update(old)
        self.assertEqual(sorted(df.model.unique()), sorted(ge.MODELS))
        self.assertTrue(ge.check(df)[0])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("**Check: PASS.**", ge.summarise(df, Path(tmp)))


if __name__ == "__main__":
    unittest.main()
