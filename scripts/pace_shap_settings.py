"""Choose PACE-SHAP's settings from approximation quality alone (before RQ1 v3).

Faithfulness is NOT looked at here. For each dataset, on seed 99 (never used
in any faithfulness run), PACE-SHAP is computed with the full 200-row
explainer background (exact when d <= 12, otherwise 128 orders per row) as
the reference. Cheaper settings use the first R rows of the same background
and, for d > 12, P orders per row. For each setting we report:

* relative L1 error  ||phi - phi_ref||_1 / ||phi_ref||_1  (mean over points);
* signed top-3 overlap with the reference (Jaccard, mean over points);
* milliseconds per explanation.

Selection rule, fixed before this script was run:
  R = the smallest of 20, 50, 100, 200 such that, with the largest P in the grid,
      every dataset has mean relative L1 error <= 0.10 and mean top-3 overlap >= 0.90;
  P = then the smallest P in the grid that meets the same bounds at that R.
  If nothing meets them, R = 200 and P = 32.

Usage:  python scripts/pace_shap_settings.py   (writes results/faithfulness_v3/settings.md)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn.explain import PaceShap  # noqa: E402
from run_faithfulness import DATASETS, backgrounds_v3, setup  # noqa: E402

SEED = 99
N_POINTS = 20
R_GRID = (20, 50, 100, 200)
P_GRID = (2, 4, 8, 16)
REF_P = 128
MAX_ERR, MIN_TOP3 = 0.10, 0.90


def top3_overlap(a, b):
    ta = np.argsort(-a, axis=1, kind="stable")[:, :3]
    tb = np.argsort(-b, axis=1, kind="stable")[:, :3]
    return float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(ta, tb)]))


def main():
    rows = []
    for name in DATASETS:
        _, Xtr, Xte, model, _ = setup(name, SEED, N_POINTS)
        bex = backgrounds_v3(Xtr, SEED)[1]
        X = Xte[:N_POINTS]
        ref = PaceShap(model, bex, n_permutations=REF_P, seed=SEED).explain(X)
        d = X.shape[1]
        exact = d <= 12
        for R in R_GRID:
            for P in ((None,) if exact else P_GRID):
                ex = PaceShap(model, bex[:R], n_permutations=P or 2, seed=SEED + 1)
                t0 = time.perf_counter()
                phi = ex.explain(X)
                ms = 1000 * (time.perf_counter() - t0) / len(X)
                err = float(np.mean(np.abs(phi - ref).sum(1) / np.maximum(np.abs(ref).sum(1), 1e-12)))
                rows.append({"dataset": name, "d": d, "mode": "exact" if exact else "sampled",
                             "R": R, "P": P, "rel_l1_error": err, "top3_overlap": top3_overlap(phi, ref),
                             "ms_per_explanation": ms})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "faithfulness_v3"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "settings.csv", index=False)

    def ok(sub):
        return bool((sub.rel_l1_error <= MAX_ERR).all() and (sub.top3_overlap >= MIN_TOP3).all())

    def at(R, P):
        sub = df[df.R == R]
        return sub[(sub["mode"] == "exact") | (sub.P == P)]

    chosen_R = next((R for R in R_GRID if ok(at(R, max(P_GRID)))), None)
    if chosen_R is None:
        chosen_R, chosen_P = 200, 32
    else:
        chosen_P = next(P for P in P_GRID if ok(at(chosen_R, P)))
    L = ["# PACE-SHAP settings (approximation quality only, seed 99)", "",
         f"Reference: 200 explainer-background rows; exact for d <= 12, otherwise {REF_P} orders per row. "
         f"{N_POINTS} test points per dataset. Faithfulness was not computed.", "",
         f"Rule (fixed before running): smallest R with every dataset at relative L1 error <= {MAX_ERR} "
         f"and top-3 overlap >= {MIN_TOP3} (at P = {max(P_GRID)}), then the smallest such P.", "",
         f"**Chosen: R = {chosen_R} background rows, P = {chosen_P} orders per row (d > 12).**", "",
         "| Dataset | d | Mode | R | P | Rel. L1 error | Top-3 overlap | ms/explanation |",
         "|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        L.append(f"| {r.dataset} | {r.d} | {r.mode} | {r.R} | {'' if pd.isna(r.P) else int(r.P)} | "
                 f"{r.rel_l1_error:.3f} | {r.top3_overlap:.2f} | {r.ms_per_explanation:.1f} |")
    (out / "settings.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"chosen R={chosen_R} P={chosen_P}")


if __name__ == "__main__":
    main()
