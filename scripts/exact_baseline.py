"""Test 1 of results/review_v1/PREREGISTRATION.md (descriptive): PACE-SHAP against
exact Shapley values computed without CFMNN's structure.

(a) Exact mode (at most 12 features). On v3's models and 200 explainer rows
    (seeds 30-32, the paper's timing seeds; 10 points each), PACE-SHAP is
    compared with shap's ExactExplainer (``xai_baselines.exact_shap``) and with
    this repository's KernelSHAP solver given every coalition
    (``kernel_shap_game`` with nsamples = 2^d). All three compute the same
    interventional Shapley values of the margin on the same rows. Reported:
    the largest absolute difference from PACE-SHAP, and milliseconds per
    explanation.
(b) Sampled mode (more than 12 features). On Wine (13 features; all 36
    points, seeds 30-32) and Letter (16; 5 points, seed 30), PACE-SHAP's
    sampled mode with 8, 16, 32 and 64 orders per row (v3 uses 32) is compared
    with exact values from PACE-SHAP's exact mode (exact_max_d = 16).
    Reported: relative L1 error and top-3 overlap.

It measures time, so run it alone, with nothing else heavy running:
    python -B scripts/exact_baseline.py
    python -B scripts/exact_baseline.py --resume        # continue after a stop
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_faithfulness as rf  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from pace_shap_settings import top3_overlap  # noqa: E402  (signed top 3, as in settings.md)

EXACT = ["S1", "S2", "S3", "shuttle", "magic", "page_blocks"]
SEEDS = [30, 31, 32]
N_EXACT = 10
SAMPLED = [("wine", 50, [30, 31, 32]), ("letter", 5, [30])]    # Wine has 36 test points
P_GRID = [8, 16, 32, 64]
OUT = ROOT / "results" / "review_v1" / "exact"
KS_ALL = "KernelSHAP, every coalition"


def rel_l1(phi, ref):
    return float(np.mean(np.abs(phi - ref).sum(axis=1) / np.maximum(np.abs(ref).sum(axis=1), 1e-12)))


def timed(fn):
    t0 = time.perf_counter()
    v = fn()
    return v, time.perf_counter() - t0


def row(part, name, seed, method, phi, ref, secs, rows, P=None):
    n, d = phi.shape
    return {"dataset": name, "seed": seed, "part": part, "method": method, "orders_per_row": P,
            "d": d, "n": n, "background_rows": rows,
            "max_abs_diff": float(np.abs(phi - ref).max()), "rel_l1_error": rel_l1(phi, ref),
            "top3_overlap": top3_overlap(phi, ref), "max_abs_value": float(np.abs(ref).max()),
            "ms_per_explanation": 1000 * secs / n}


def run_one(name, seed, n, part):
    _, Xtr, Xte, model, _ = rf.setup(name, seed, n)
    _, bex = rf.backgrounds_v3(Xtr, seed)
    B = bex[:rf.SHAP_V3_R]
    d = Xte.shape[1]
    if part == "exact":
        ref, t = timed(lambda: PaceShap(model, B).explain(Xte))
        rows = [row(part, name, seed, "PACE-SHAP", ref, ref, t, len(B))]
        for m, fn in [("shap ExactExplainer", lambda: xb.exact_shap(model, Xte, B)),
                      (KS_ALL, lambda: xb.kernel_shap_game(model, Xte, B, nsamples=2 ** d, seed=seed))]:
            phi, t = timed(fn)
            rows.append(row(part, name, seed, m, phi, ref, t, len(B)))
        print(f"{name} seed {seed}: largest difference {max(r['max_abs_diff'] for r in rows):.1e}; ms "
              + ", ".join(f"{r['method']} {r['ms_per_explanation']:.0f}" for r in rows), flush=True)
    else:
        ref, t = timed(lambda: PaceShap(model, B, exact_max_d=16).explain(Xte))
        rows = [row(part, name, seed, "PACE-SHAP, exact (reference)", ref, ref, t, len(B))]
        for P in P_GRID:
            phi, t = timed(lambda: PaceShap(model, B, n_permutations=P, seed=seed).explain(Xte))
            rows.append(row(part, name, seed, "PACE-SHAP, sampled", phi, ref, t, len(B), P))
        print(f"{name} seed {seed}: relative L1 error "
              + ", ".join(f"P={r['orders_per_row']} {r['rel_l1_error']:.4f}" for r in rows[1:])
              + f" (exact reference {rows[0]['ms_per_explanation']:.0f} ms)", flush=True)
    return rows


def summarise(df, out: Path) -> str:
    L = ["# Review Test 1 (descriptive): PACE-SHAP vs exact Shapley values computed in general ways",
         "", "Rules: [`../PREREGISTRATION.md`](../PREREGISTRATION.md) (Test 1). v3's models and explainer "
         "rows. Milliseconds per explanation, measured with nothing else running.", ""]
    ex = df[df.part == "exact"]
    if len(ex):
        L += ["## (a) Exact mode: the same values, computed three ways", "",
              "| Dataset | d | Rows | Points | Largest difference: ExactExplainer | Largest difference: "
              "KernelSHAP, every coalition | Largest value | ms PACE-SHAP | ms ExactExplainer | "
              "ms KernelSHAP, every coalition | ExactExplainer / PACE-SHAP | KernelSHAP / PACE-SHAP |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for ds, g in ex.groupby("dataset", sort=False):
            ms = g.groupby("method").ms_per_explanation.mean()
            dif = g.groupby("method").max_abs_diff.max()
            L.append(f"| {ds} | {g.d.iloc[0]} | {g.background_rows.iloc[0]} | "
                     f"{g[g.method == 'PACE-SHAP'].n.sum()} | {dif['shap ExactExplainer']:.1e} | "
                     f"{dif[KS_ALL]:.1e} | {g.max_abs_value.max():.3f} | {ms['PACE-SHAP']:.1f} | "
                     f"{ms['shap ExactExplainer']:.1f} | {ms[KS_ALL]:.1f} | "
                     f"{ms['shap ExactExplainer'] / ms['PACE-SHAP']:.1f}× | "
                     f"{ms[KS_ALL] / ms['PACE-SHAP']:.1f}× |")
        L += ["", "Largest difference: the largest absolute difference from PACE-SHAP over every point, "
              "feature and seed. Largest value: the largest absolute PACE-SHAP value, for scale."]
    sa = df[df.part == "sampled"]
    if len(sa):
        L += ["", "## (b) Sampled mode: error against exact values", "",
              "| Dataset | d | Orders per row | Points | Relative L1 error | Top-3 overlap | "
              "ms per explanation |", "|---|---|---|---|---|---|---|"]
        for ds, g in sa.groupby("dataset", sort=False):
            r0 = g[g.orders_per_row.isna()]
            L.append(f"| {ds} | {g.d.iloc[0]} | exact (reference) | {r0.n.sum()} | 0 | 1 | "
                     f"{r0.ms_per_explanation.mean():.1f} |")
            for P, h in g[g.orders_per_row.notna()].groupby("orders_per_row"):
                L.append(f"| {ds} | {g.d.iloc[0]} | {int(P)} | {h.n.sum()} | {h.rel_l1_error.mean():.4f} "
                         f"| {h.top3_overlap.mean():.3f} | {h.ms_per_explanation.mean():.1f} |")
        L += ["", "Relative L1 error: sum over features of |sampled - exact|, divided by the sum of "
              "|exact|, averaged over points. Top-3 overlap: Jaccard overlap of the top 3 features by "
              "signed value, as in `faithfulness_v3/settings.md`."]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Review Test 1: PACE-SHAP vs exact Shapley values")
    ap.add_argument("--datasets", nargs="+", help="only these datasets")
    ap.add_argument("--seeds", nargs="+", type=int, help="these seeds instead (smoke runs only)")
    ap.add_argument("--n", type=int, help="points per dataset and seed (smoke runs only)")
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true", help="continue a stopped run")
    ap.add_argument("--fresh", action="store_true", help="delete earlier results and start over")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if not a.summary_only:
        tasks = ([(d, s, N_EXACT, "exact") for d in EXACT for s in a.seeds or SEEDS]
                 + [(d, s, n, "sampled") for d, n, seeds in SAMPLED for s in a.seeds or seeds])
        tasks = [(d, s, a.n or n, p) for d, s, n, p in tasks if not a.datasets or d in a.datasets]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, jobs=1)   # alone: it is timed
    print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
