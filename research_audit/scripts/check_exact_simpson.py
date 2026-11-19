"""Run I, post hoc: which side carries the 1e-10 differences of the exact check?

Run I's exact check (results/generality/exact.csv) passed its limit of 1e-9 on every task, but
5 of 60 tasks show differences of 7e-11 to 1.6e-10 (S1 152, S2 159, S3 150 and 159, MAGIC 158);
the other 55 are near 1e-16. This script refits the worst case(s) exactly as `run_exact` in
scripts/run_generality.py does, and compares PACE-SHAP and shap's ExactExplainer with a
brute-force Shapley value: every coalition S evaluated on every explainer row by calling the
model (`xai_baselines.margin_score`), the game value v(S) summed with math.fsum, and
phi_i = sum_S w(S) [v(S + i) - v(S)] summed with math.fsum.

It also counts hybrid inputs whose top two class supports are within the model's tie
tolerance, where a rounding-level change in a membership can flip the nearest-centroid rule.

Post hoc and read-only: it writes nothing under results/. Run from the repository root:
    python -B research_audit/scripts/check_exact_simpson.py [dataset seed ...] > research_audit/scripts/check_exact_simpson_out.txt
"""

from __future__ import annotations

import itertools
import math
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
sys.stdout.reconfigure(encoding="utf-8")

import run_faithfulness as rf  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.contraction import contract_all  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402

N_EXACT = 5


def brute_force(model, x, B, a):
    """Exact interventional Shapley values of class a's margin at x, on rows B, with fsum."""
    d = len(x)
    v = {}
    for k in range(d + 1):
        for S in itertools.combinations(range(d), k):
            Z = B.copy()
            Z[:, list(S)] = x[list(S)]
            v[S] = math.fsum(xb.margin_score(model, Z, a)) / len(B)
    phi = np.zeros(d)
    for i in range(d):
        terms = []
        rest = [j for j in range(d) if j != i]
        for k in range(d):
            w = math.factorial(k) * math.factorial(d - k - 1) / math.factorial(d)
            for S in itertools.combinations(rest, k):
                Si = tuple(sorted(S + (i,)))
                terms.append(w * (v[Si] - v[S]))
        phi[i] = math.fsum(terms)
    return phi


def near_ties(model, x, B):
    """Hybrid inputs (all coalitions) whose best and second-best supports differ by <= the tie tolerance."""
    d, tol = len(x), getattr(model, "tie_tol", 1e-9)
    n = close = 0
    for k in range(d + 1):
        for S in itertools.combinations(range(d), k):
            Z = B.copy()
            Z[:, list(S)] = x[list(S)]
            s = np.sort(model.predict_details(Z).support_rescaled, axis=1)
            gap = s[:, -1] - s[:, -2]
            n += len(Z)
            close += int(np.sum((gap > 0) & (gap <= 100 * tol)))
    return close, n


def one(name, seed):
    warnings.filterwarnings("ignore")
    _, Xtr, Xte, model, _ = rf.setup(name, seed, N_EXACT)
    _, bex = rf.backgrounds_v3(Xtr, seed)
    B = bex[:rf.SHAP_V3_R]
    sim, st, _ = contract_all(model, Xtr)
    pace = PaceShap(sim, B).explain(Xte)
    ref = xb.exact_shap(sim, Xte, B)
    diff = np.abs(pace - ref)
    p, f = np.unravel_index(diff.argmax(), diff.shape)
    print(f"\n{name} seed {seed}: {st['contractions']} contractions, {sim.n_boxes_} boxes")
    print(f"  largest |PACE-SHAP - ExactExplainer| = {diff.max():.3e} at point {p}, feature {f}")
    print(f"  per point: {[f'{v:.1e}' for v in diff.max(axis=1)]}")
    a = int(sim.predict_details(Xte[p:p + 1]).cls[0])
    bf = brute_force(sim, Xte[p], B, a)
    print(f"  brute force (fsum) vs PACE-SHAP:      largest {np.abs(bf - pace[p]).max():.3e}, at feature {f}: {abs(bf[f] - pace[p, f]):.3e}")
    print(f"  brute force (fsum) vs ExactExplainer: largest {np.abs(bf - ref[p]).max():.3e}, at feature {f}: {abs(bf[f] - ref[p, f]):.3e}")
    print(f"  efficiency gaps (sum phi - [v(all) - v(empty)]): PACE {pace[p].sum() - bf.sum():+.3e}, "
          f"Exact {ref[p].sum() - bf.sum():+.3e}")
    close, n = near_ties(sim, Xte[p], B)
    print(f"  hybrid inputs whose top two supports are within 100 x tie_tol: {close} of {n}")


def main():
    args = sys.argv[1:] or ["S3", "150"]
    for name, seed in zip(args[::2], args[1::2]):
        one(name, int(seed))


if __name__ == "__main__":
    main()
