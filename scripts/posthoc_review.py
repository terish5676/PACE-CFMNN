"""Post hoc checks from the review of the RQ1 extensions (not pre-registered).

1. Interactions (B4): how much (x0, x1) stands out from the next strongest
   pair on S1, S2 and S3 (S1 and S2 have no true interaction, so they act as
   a negative control for the S3 test). Reads the B4 runs.csv; no models.
2. Conditional game (B3): the effective number of explainer rows the
   Gaussian weights use at sigma = 0.1 (Kish: (sum w)^2 / sum w^2), for
   coalitions of several sizes. Seed 40, the 50 test points and the 200
   explainer rows of the cond protocol; no models.
3. Global importance (B2, optional ``--points DATASET``): PACE-flip averaged
   over all test points (as in B2) or over the same 50 points mean
   |KernelSHAP| uses, each against permutation importance (recomputed with
   B2's exact call, so it matches B2's numbers). Trains one model per seed.

Writes ``results/extensions_v1/posthoc_review.md`` (and, with ``--points``,
``results/extensions_v1/global_importance/posthoc_points_<dataset>.md``).

Usage (from the project folder):
    python scripts/posthoc_review.py
    python scripts/posthoc_review.py --points page_blocks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.explain import PaceFlip, cond_scale  # noqa: E402
from run_faithfulness import COND_DATASETS, COND_SIGMA, SHAP_V3_R, backgrounds_v3  # noqa: E402

OUT = ROOT / "results" / "extensions_v1"


def dominance() -> list[str]:
    d = pd.read_csv(OUT / "interactions" / "runs.csv")
    L = ["## 1. Interactions: does (x0, x1) stand out more on S3 than on S1 and S2?", "",
         "S1 and S2 add their features up (no true interaction); S3 has the AND of x0 and x1. Per seed: "
         "the strength of (x0, x1) divided by the next strongest pair's, and its share of all pair "
         "strength. Mean (min–max) over seeds 40–49.", "",
         "| Dataset | (x0, x1) / next strongest pair | Share of all pair strength |", "|---|---|---|"]
    for ds in ["S1", "S2", "S3", "S1c", "S2c"]:
        ratio, share = [], []
        for _, h in d[d.dataset == ds].groupby("seed"):
            pair = (h.i == 0) & (h.j == 1)
            a = h[pair].strength.iloc[0]
            ratio.append(a / h[~pair].strength.max())
            share.append(a / h.strength.sum())
        r = np.array(ratio)
        L.append(f"| {ds} | {r.mean():.2f} ({r.min():.2f}–{r.max():.2f}) | {np.mean(share):.3f} |")
    return L


def split(name, seed):
    ds = data.load(name, seed=seed)
    sp = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])
    sc = data.fit_scaler(ds.X[order])
    return ds, sp, sc, order


def ess(seed: int = 40, n_coal: int = 40) -> list[str]:
    L = ["", f"## 2. Conditional game: effective number of explainer rows (sigma = {COND_SIGMA})", "",
         f"Seed {seed}; the cond protocol's 50 test points and {SHAP_V3_R} explainer rows. For each "
         f"point and coalition size, {n_coal} random coalitions; row weights exp(-D_S^2 / (2 sigma^2)) "
         "as in PaceShap(sigma=...). Effective rows = (sum w)^2 / sum w^2 (Kish), median over points "
         "and coalitions. 1 means a single row carries all the weight.", "",
         "| Dataset | Features | |S| = 1 | 2 | d/4 | d/2 | 3d/4 | d - 1 |", "|---|---|---|---|---|---|---|---|"]
    for name in COND_DATASETS:
        ds, sp, sc, order = split(name, seed)
        Xtr = sc.transform(ds.X[order])
        te = np.random.default_rng(seed).permutation(sp["test"])[:50]
        Xte = sc.transform(ds.X[te])
        B = backgrounds_v3(Xtr, seed)[1][:SHAP_V3_R]
        scale = cond_scale(B)
        d = B.shape[1]
        rng = np.random.default_rng(0)
        cells = []
        for t in [1, 2, max(1, d // 4), max(1, d // 2), max(1, (3 * d) // 4), d - 1]:
            vals = []
            for x in Xte:
                q = ((x - B) / scale) ** 2
                for _ in range(n_coal):
                    lw = -q[:, rng.choice(d, t, replace=False)].mean(axis=1) / (2 * COND_SIGMA ** 2)
                    w = np.exp(lw - lw.max())
                    vals.append(w.sum() ** 2 / (w ** 2).sum())
            cells.append(f"{np.median(vals):.1f}")
        L.append(f"| {name} | {d} | " + " | ".join(cells) + " |")
    return L


def points(name: str, seeds=range(40, 50)) -> list[str]:
    from sklearn.inspection import permutation_importance
    runs = pd.read_csv(OUT / "global_importance" / "runs.csv")
    rows = []
    for seed in seeds:
        ds, sp, sc, order = split(name, seed)
        Xtr, ytr = sc.transform(ds.X[order]), ds.y[order]
        Xte, yte = np.clip(sc.transform(ds.X[sp["test"]]), 0.0, 1.0), ds.y[sp["test"]]
        model = CFMNN().fit(Xtr, ytr)
        bg = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(200, len(Xtr)), replace=False)]
        pts = Xte[np.random.default_rng(seed).permutation(len(Xte))[:50]]
        perm = permutation_importance(model, Xte, yte, n_repeats=10, random_state=seed,
                                      scoring="accuracy").importances_mean
        pf = PaceFlip(model, bg)
        all_rows, fifty = pf.explain(Xte).mean(axis=0), pf.explain(pts).mean(axis=0)
        old = runs[(runs.dataset == name) & (runs.seed == seed)].set_index("method")
        rows.append(dict(seed=seed, n_test=len(Xte),
                         all_points=stats.spearmanr(all_rows, perm)[0],
                         all_points_in_b2=old.loc["PACE-flip, averaged", "rho_Permutation importance"],
                         same_50=stats.spearmanr(fifty, perm)[0],
                         kernelshap_in_b2=old.loc["Mean abs KernelSHAP", "rho_Permutation importance"],
                         same_50_vs_all=stats.spearmanr(fifty, all_rows)[0]))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    gap50 = df.same_50 - df.kernelshap_in_b2
    p = stats.wilcoxon(gap50).pvalue if (gap50 != 0).any() else 1.0
    L = [f"# Post hoc: PACE-flip's global ranking on all test points vs the same 50 points ({name})", "",
         "Not pre-registered. Spearman rho with permutation importance (recomputed with B2's exact call). "
         "PACE-flip averaged over all test points (as in B2) or over the 50 points mean |KernelSHAP| "
         "uses. Generated by `scripts/posthoc_review.py --points " + name + "`.", "",
         "| Seed | Test points | PACE-flip, all points | (same, in B2's runs.csv) | PACE-flip, same 50 points | "
         "Mean abs KernelSHAP (B2) | PACE-flip 50 vs all |", "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {int(r.seed)} | {int(r.n_test)} | {r.all_points:.3f} | {r.all_points_in_b2:.3f} | "
                 f"{r.same_50:.3f} | {r.kernelshap_in_b2:.3f} | {r.same_50_vs_all:.3f} |")
    L += ["", f"- Means: PACE-flip all points {df.all_points.mean():.3f}; same 50 points "
          f"{df.same_50.mean():.3f}; mean abs KernelSHAP {df.kernelshap_in_b2.mean():.3f}.",
          f"- PACE-flip (same 50 points) minus mean abs KernelSHAP: {gap50.mean():+.3f}, higher in "
          f"{int((gap50 > 0).sum())} of {len(df)} seeds, Wilcoxon p = {p:.3g} (this dataset alone, no "
          "correction).",
          f"- PACE-flip's 50-point ranking agrees with its all-points ranking at "
          f"{df.same_50_vs_all.mean():.2f} on average."]
    text = "\n".join(L) + "\n"
    (OUT / "global_importance" / f"posthoc_points_{name}.md").write_text(text, encoding="utf-8")
    return L


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="*", default=[], help="datasets for check 3 (trains models)")
    a = ap.parse_args()
    L = ["# Post hoc checks from the review of the RQ1 extensions (not pre-registered)", "",
         "Generated by `scripts/posthoc_review.py`. These were decided after the pre-registered runs, "
         "so they describe and qualify the results; they test nothing.", ""]
    L += dominance() + ess()
    text = "\n".join(L) + "\n"
    (OUT / "posthoc_review.md").write_text(text, encoding="utf-8")
    print(text)
    for name in a.points:
        print("\n".join(points(name)))


if __name__ == "__main__":
    main()
