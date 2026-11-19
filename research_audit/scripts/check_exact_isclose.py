"""Run I, post hoc: what inside shap's ExactExplainer causes the 1e-10 differences?

`check_exact_simpson.py` showed that in Run I's exact check (results/generality/exact.csv) the
differences of 7e-11 to 1.6e-10 come from shap's ExactExplainer, not from PACE-SHAP. This script
finds the cause.

shap 0.54.1's tabular masker (`shap.maskers.Independent`) has `invariants(x)`, which returns
`np.isclose(x, background)` (relative tolerance 1e-5, absolute 1e-8). For a coalition step,
ExactExplainer re-evaluates only the background rows where a changed feature is not
"invariant"; the other rows keep their previous output. So a background value close to, but not
equal to, the explained point's value is treated as equal, and the model is not called on it.

Part 1: for every one of the 60 tasks, the number of (point, explainer row, feature) triples
with np.isclose true but the values unequal, beside the committed largest difference.
Part 2: for each task whose committed difference is above 1e-14, at the point that holds such a
triple, PACE-SHAP against ExactExplainer as shipped, and against ExactExplainer with exact
equality in place of np.isclose (the method is replaced in this process only; nothing on disk
changes).

Each task is rebuilt exactly as `run_exact` in scripts/run_generality.py does. Post hoc and
read-only: it writes nothing under results/. Run from the repository root:
    python -B research_audit/scripts/check_exact_isclose.py > research_audit/scripts/check_exact_isclose_out.txt
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

import run_faithfulness as rf  # noqa: E402
import shap  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.contraction import contract_all  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402

N_EXACT = 5           # run_generality.N_EXACT
SMALL = 1e-14         # committed differences above this are the five to explain


def task(name, seed):
    _, Xtr, Xte, model, _ = rf.setup(name, seed, N_EXACT)
    _, bex = rf.backgrounds_v3(Xtr, seed)
    return Xtr, Xte, model, bex[:rf.SHAP_V3_R]


def near_equal(x, B):
    """(explainer row, feature) pairs that shap treats as unchanged although they differ."""
    c = np.isclose(x, B) & (x != B)
    return int(c.sum()), float(np.abs(x - B)[c].max()) if c.any() else 0.0


def check(label, ok):
    print(f"{'OK ' if ok else 'BAD'} {label}")


print("shap", shap.__version__)
E = pd.read_csv(ROOT / "results" / "generality" / "exact.csv")
print(f"\nPart 1: near-equal values in each of the {len(E)} tasks")
print("dataset      seed  committed diff  near-equal pairs per point   largest gap")
rows = []
for _, r in E.iterrows():
    _, Xte, _, B = task(r.dataset, int(r.seed))
    per = [near_equal(x, B) for x in Xte]
    rows.append({"dataset": r.dataset, "seed": int(r.seed), "diff": float(r.max_abs_diff),
                 "pairs": [n for n, _ in per], "gap": max(g for _, g in per)})
    print(f"{r.dataset:12s} {int(r.seed)}  {r.max_abs_diff:14.2e}  {str(rows[-1]['pairs']):27s}"
          f"  {rows[-1]['gap']:.1e}", flush=True)
R = pd.DataFrame(rows)
R["n"] = R.pairs.map(sum)
big, small = R[R["diff"] > SMALL], R[R["diff"] <= SMALL]
print(f"\ntasks above {SMALL:g}: {len(big)}; each has exactly one near-equal pair: "
      f"{bool((big.n == 1).all())}; gaps {big.gap.min():.1e} to {big.gap.max():.1e}")
print(f"tasks at or below {SMALL:g}: {len(small)}; largest committed difference "
      f"{small['diff'].max():.1e}; with a near-equal pair: "
      f"{list(zip(small[small.n > 0].dataset, small[small.n > 0].seed))}")
check("5 tasks above 1e-14", len(big) == 5)
check("each of them has exactly one near-equal pair", bool((big.n == 1).all()))
check("the other 55 are at most 2.7e-16", bool(small["diff"].max() <= 2.75e-16))

print("\nPart 2: the point holding the near-equal pair, ExactExplainer with and without np.isclose")
shipped = shap.maskers.Independent.invariants
worst = []
for _, r in big.iterrows():
    Xtr, Xte, model, B = task(r.dataset, int(r.seed))
    sim, _, _ = contract_all(model, Xtr)
    p = int(np.argmax(r.pairs))
    x = Xte[p:p + 1]
    phi = PaceShap(sim, B).explain(x)[0]
    shap.maskers.Independent.invariants = shipped
    as_shipped = xb.exact_shap(sim, x, B)[0]
    shap.maskers.Independent.invariants = lambda self, x: x == self.data
    exact_eq = xb.exact_shap(sim, x, B)[0]
    shap.maskers.Independent.invariants = shipped
    a, b = np.abs(phi - as_shipped).max(), np.abs(phi - exact_eq).max()
    worst.append((a, b))
    print(f"{r.dataset:6s} seed {r.seed} point {p}: |PACE - shipped| {a:.3e} "
          f"(committed {r['diff']:.3e}); |PACE - exact equality| {b:.3e}", flush=True)
a = np.array(worst)
print(f"\nas shipped: {a[:, 0].min():.1e} to {a[:, 0].max():.1e}; "
      f"with exact equality: at most {a[:, 1].max():.1e}")
check("as shipped, each point reproduces its task's committed difference",
      bool(np.allclose(a[:, 0], big["diff"].to_numpy(), rtol=0.01)))
check("with exact equality, every difference is below 1e-15", bool(a[:, 1].max() < 1e-15))
