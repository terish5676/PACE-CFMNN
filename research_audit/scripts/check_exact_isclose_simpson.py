"""Run M, post hoc: is the one 1e-10 difference in its exact check shap's np.isclose shortcut?

Run M (results/simpson_fmm/exact.csv) compared PACE-SHAP with shap's ExactExplainer on Simpson's
classifier trained from scratch. The check passed, but one task (S1, seed 222) differs by 1.4e-10
while the other 59 agree to 3.1e-16. `check_exact_isclose.py` traced Run I's differences of the
same size to shap 0.54.1's tabular masker: `Independent.invariants(x)` returns
`np.isclose(x, background)` (relative tolerance 1e-5, absolute 1e-8), and ExactExplainer does not
call the model again for a background row whose only changed features are "invariant". This
script repeats that check on Run M's tasks.

Part 1: for every one of the 60 tasks, the number of (explainer row, feature) pairs per point with
np.isclose true but the values unequal, beside the committed largest difference.
Part 2: for each task whose committed difference is above 1e-14, at the point that holds such a
pair, PACE-SHAP against ExactExplainer as shipped, and against ExactExplainer with exact equality
in place of np.isclose (the method is replaced in this process only; nothing on disk changes).

Each task is rebuilt exactly as `run_exact` in scripts/run_simpson_fmm.py does (its own
`exact_inputs`, then `SimpsonFMM(theta=0.3, gamma=4)`). Post hoc and read-only: it writes nothing
under results/. Run from the repository root:
    python -B research_audit/scripts/check_exact_isclose_simpson.py > research_audit/scripts/check_exact_isclose_simpson_out.txt
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import run_simpson_fmm as rm  # noqa: E402
import shap  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.simpson_fmm import SimpsonFMM  # noqa: E402

SMALL = 1e-14         # committed differences above this are the ones to explain


def near_equal(x, B):
    """(explainer row, feature) pairs that shap treats as unchanged although they differ."""
    c = np.isclose(x, B) & (x != B)
    return int(c.sum()), float(np.abs(x - B)[c].max()) if c.any() else 0.0


def check(label, ok):
    print(f"{'OK ' if ok else 'BAD'} {label}")


print("shap", shap.__version__)
E = pd.read_csv(ROOT / "results" / "simpson_fmm" / "exact.csv", float_precision="round_trip")
print(f"\nPart 1: near-equal values in each of the {len(E)} tasks")
print("dataset      seed  committed diff  near-equal pairs per point   largest gap")
rows = []
for _, r in E.iterrows():
    _, _, Xte, B = rm.exact_inputs(r.dataset, int(r.seed))
    per = [near_equal(x, B) for x in Xte]
    rows.append({"dataset": r.dataset, "seed": int(r.seed), "diff": float(r.max_abs_diff),
                 "pairs": [n for n, _ in per], "gap": max(g for _, g in per)})
    print(f"{r.dataset:12s} {int(r.seed)}  {r.max_abs_diff:14.2e}  {str(rows[-1]['pairs']):27s}"
          f"  {rows[-1]['gap']:.1e}", flush=True)
R = pd.DataFrame(rows)
R["n"] = R.pairs.map(sum)
big, small = R[R["diff"] > SMALL], R[R["diff"] <= SMALL]
print(f"\ntasks above {SMALL:g}: {len(big)} ({list(zip(big.dataset, big.seed))}); near-equal pairs "
      f"in them: {big.n.tolist()}; gaps {big.gap.tolist()}")
print(f"tasks at or below {SMALL:g}: {len(small)}; largest committed difference "
      f"{small['diff'].max():.1e}; with a near-equal pair: "
      f"{list(zip(small[small.n > 0].dataset, small[small.n > 0].seed))}")
check("1 task above 1e-14 (S1, seed 222)", list(zip(big.dataset, big.seed)) == [("S1", 222)])
check("it has exactly one near-equal pair", bool((big.n == 1).all()))
check("the other 59 are at most 3.1e-16", bool(small["diff"].max() <= 3.1e-16))

print("\nPart 2: the point holding the near-equal pair, ExactExplainer with and without np.isclose")
shipped = shap.maskers.Independent.invariants
worst = []
for _, r in big.iterrows():
    Xtr, ytr, Xte, B = rm.exact_inputs(r.dataset, int(r.seed))
    sim = SimpsonFMM(theta=rm.THETA, gamma=rm.GAMMA).fit(Xtr, ytr)
    p = int(np.argmax(r.pairs))
    x = Xte[p:p + 1]
    ex = PaceShap(sim, B)
    phi = ex.explain(x)[0]
    shap.maskers.Independent.invariants = shipped
    as_shipped = xb.exact_shap(sim, x, B)[0]
    shap.maskers.Independent.invariants = lambda self, x: x == self.data
    exact_eq = xb.exact_shap(sim, x, B)[0]
    shap.maskers.Independent.invariants = shipped
    a, b = np.abs(phi - as_shipped).max(), np.abs(phi - exact_eq).max()
    worst.append((a, b))
    print(f"{r.dataset:6s} seed {r.seed} point {p} ({sim.n_boxes_} boxes, exact mode {ex.exact}): "
          f"|PACE - shipped| {a:.3e} (committed {r['diff']:.3e}); |PACE - exact equality| {b:.3e}",
          flush=True)
a = np.array(worst)
print(f"\nas shipped: {a[:, 0].max():.2e}; with exact equality: {a[:, 1].max():.2e}")
check("as shipped, the point reproduces the task's committed difference",
      bool(np.allclose(a[:, 0], big["diff"].to_numpy(), rtol=0.01)))
check("with exact equality, the difference is below 1e-15", bool(a[:, 1].max() < 1e-15))
