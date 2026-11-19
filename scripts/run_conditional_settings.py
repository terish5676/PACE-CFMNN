"""Run D: B3's conditional test at more settings of sigma and of the number of explainer rows.

Rules: results/conditional_settings/PREREGISTRATION.md. Plan: research_audit/LIMITATIONS_PLAN.md, run D.

B3 (`scripts/run_faithfulness.py`, `run_cond`; results/extensions_v1/conditional) ran the conditional
Shapley game once: sigma = 0.1 with 200 explainer rows. This run varies both, on the same design:

* settings (sigma, rows): (0.05, 200), (0.2, 200), (0.4, 200), (0.2, 800);
* explainers: PACE-SHAP-cond and KernelSHAP-cond (200 coalitions, as in B3), on the same rows;
* datasets S1, Shuttle, Letter, Landsat, MAGIC and Page Blocks, 50 test points per seed.

The explainer rows are B3's own 200 rows followed by further training rows, never the evaluation
rows (`posthoc_rows.explainer_rows`); the weights' per-feature scale is B3's, fixed from the first
200 rows (as `posthoc_rows.py` does), so only sigma and the number of rows change. The evaluation
rows and the four tests (interventional and conditional deletion and insertion margin AUCs) are
B3's. Per setting and method: the effective number of rows of the conditional weights, and the share
of the game's inputs that lie off the data (`pace_cfmnn/realism.py`).

    python -B scripts/run_conditional_settings.py --jobs 12
    python -B scripts/run_conditional_settings.py --jobs 12 --resume
    python -B scripts/run_conditional_settings.py --summary-only
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

from pace_cfmnn import realism  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import PaceShap, cond_scale  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from posthoc_rows import explainer_rows  # noqa: E402
from posthoc_sigma import game_off_share  # noqa: E402
from run_error_detection import ci95, paired  # noqa: E402
from run_faithfulness import (COND_NSAMPLES, SHAP_V3_P, SHAP_V3_R, SYNTHETIC, backgrounds_v3,  # noqa: E402
                              cond_curves, deletion_auc, insertion_auc, setup, top3_random_ties)

OUT = ROOT / "results" / "conditional_settings"
DATASETS = ["S1", "shuttle", "letter", "landsat", "magic", "page_blocks"]
SEEDS = list(range(110, 120))
N_POINTS = 50
SETTINGS = [(0.05, 200), (0.2, 200), (0.4, 200), (0.2, 800)]
METHODS = ["PACE-SHAP-cond", "KernelSHAP-cond"]
N_COALITIONS = 100                      # sampled coalitions per point for the effective rows
# (metric, lower is better): the pre-registered primaries, for the -cond pair at each setting
PRIMARIES = [("cond_margin_auc", True), ("cond_insertion_margin_auc", False)]
COST = {"letter": 6, "landsat": 5, "magic": 4, "shuttle": 3, "page_blocks": 3, "S1": 1}


def label(sigma: float, rows: int) -> str:
    return f"sigma={sigma:g}, rows={rows}"


def effective_rows_sampled(X, B, scale, sigma, rng, k=N_COALITIONS):
    """Kish effective rows (sum w)^2 / sum w^2 of the conditional weights exp(-D_S^2 / (2 sigma^2)),
    median over points and over ``k`` random coalitions per point: (coalitions with at least half
    the features, smaller non-empty coalitions). Works for any number of features."""
    n, d = X.shape
    S = realism._coalitions(n, d, k, rng)                          # (n, k, d)
    large, small = [], []
    for p in range(n):
        q = ((X[p] - B) / scale) ** 2                              # (R, d)
        size = S[p].sum(axis=1)
        logw = -(S[p].astype(float) @ q.T) / (2.0 * sigma ** 2 * np.maximum(size, 1)[:, None])
        w = np.exp(logw - logw.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        kish = 1.0 / (w ** 2).sum(axis=1)
        ok = (size >= 1) & (size < d)
        large.append(kish[ok & (size >= d / 2)])
        small.append(kish[ok & (size < d / 2)])
    med = lambda parts: float(np.median(np.concatenate(parts))) if any(len(a) for a in parts) else np.nan  # noqa: E731
    return med(large), med(small)


def run_one(name: str, seed: int, n: int = N_POINTS, settings=SETTINGS):
    t_start = time.perf_counter()
    ds, Xtr, Xte, model, _ = setup(name, seed, n)
    bg, bex = backgrounds_v3(Xtr, seed)
    scale = cond_scale(bex[:SHAP_V3_R])            # B3's weight scale, kept for every setting
    relevant = set(ds.info.get("relevant", [])) if name in SYNTHETIC else None
    man = realism.Manifold(Xtr, seed=seed)
    rows = []
    for sigma, R in settings:
        B = explainer_rows(Xtr, seed, R)
        if not np.array_equal(B[:min(R, SHAP_V3_R)], bex[:min(R, SHAP_V3_R)]):
            raise AssertionError("the explainer rows do not start with B3's")
        k_large, k_small = effective_rows_sampled(Xte, B, scale, sigma,
                                                  np.random.default_rng(seed + 1900))
        off = game_off_share(man, Xte, B, scale, seed, sigma)
        fns = {"PACE-SHAP-cond": lambda: PaceShap(model, B, n_permutations=SHAP_V3_P, seed=seed,
                                                  sigma=sigma, scale=scale).explain(Xte),
               "KernelSHAP-cond": lambda: xb.kernel_shap_game(model, Xte, B, nsamples=COND_NSAMPLES,
                                                              seed=seed, sigma=sigma, scale=scale)}
        for m in METHODS:
            t0 = time.perf_counter()
            attr = fns[m]()
            dt = (time.perf_counter() - t0) / len(Xte)
            _, del_int = deletion_auc(model, Xte, attr, bg, seed=seed, signed=True)
            _, ins_int = insertion_auc(model, Xte, attr, bg, seed=seed)
            _, del_cond = cond_curves(model, Xte, attr, bg, seed=seed)
            _, ins_cond = cond_curves(model, Xte, attr, bg, seed=seed, insertion=True)
            t3 = top3_random_ties(attr, np.random.default_rng(seed + 7))
            rows.append({"dataset": name, "seed": seed, "setting": label(sigma, R), "sigma": sigma,
                         "rows": R, "method": m, "eff_rows_large": k_large, "eff_rows_small": k_small,
                         "off_share": off, "margin_auc": del_int.mean(),
                         "insertion_margin_auc": ins_int.mean(), "cond_margin_auc": del_cond.mean(),
                         "cond_insertion_margin_auc": ins_cond.mean(),
                         "gt_top3_precision": (np.mean([len(set(t) & relevant) / 3 for t in t3])
                                               if relevant else np.nan),
                         "ms_per_explanation": 1000 * dt, "n": len(Xte)})
    print(f"{name} seed {seed}: {len(Xte)} points, {time.perf_counter() - t_start:.0f} s", flush=True)
    return rows


# ------------------------------------------------------------------ summary
def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    ds = [d for d in DATASETS if d in set(df.dataset)]
    settings = [label(*s) for s in SETTINGS if label(*s) in set(df.setting)]
    f = lambda v: f"{v.mean():.3f} ± {ci95(v):.3f}"   # noqa: E731
    L = ["# Run D: the conditional test at more settings", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). B3's design, seeds "
         f"{seeds[0]}–{seeds[-1]} ({len(seeds)}), {int(df.n.max())} test points per dataset and "
         "seed. Conditional deletion margin AUC: lower is more faithful; conditional insertion "
         "margin AUC: higher. Each comparison: paired over seeds, two-sided Wilcoxon signed-rank "
         "test, Holm across datasets. Written by `scripts/run_conditional_settings.py`.", "",
         "## Primary: PACE-SHAP-cond vs KernelSHAP-cond at each setting", ""]
    for setting in settings:
        sub = df[df.setting == setting]
        for metric, lower in PRIMARIES:
            res = paired(sub, "PACE-SHAP-cond", "KernelSHAP-cond", metric=metric)
            good = res["diff"] < 0 if lower else res["diff"] > 0
            sig = res["p_holm"] < 0.05
            L += [f"**{setting}, {metric}** ({'negative' if lower else 'positive'} favours PACE-SHAP-cond): "
                  f"better on {int((sig & good).sum())}, level on {int((~sig).sum())}, worse on "
                  f"{int((sig & ~good).sum())} of {len(res)} datasets.", "",
                  "| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) | Verdict |",
                  "|---|---|---|---|---|---|"]
            for (_, r), g, s in zip(res.iterrows(), good, sig):
                L.append(f"| {r.dataset} | {r['diff']:+.4f} ± {r.ci:.4f} | {r.wins} of {r.n} | "
                         f"{r.p:.3g} | {r.p_holm:.3g} | {'level' if not s else 'better' if g else 'worse'} |")
            L.append("")
    L += ["## Effective rows and inputs off the data, by setting (mean over seeds)", "",
          "Effective rows: Kish (Σw)² / Σw² of the conditional weights, median over points and "
          f"{N_COALITIONS} random coalitions per point, for coalitions with at least half the "
          "features (large) and smaller ones. Off the data: share of the game's inputs farther "
          "from the training rows than their own nearest-neighbour distances (B3's measure).", "",
          "| Dataset | Setting | Effective rows (large) | Effective rows (small) | Off the data |",
          "|---|---|---|---|---|"]
    one = df[df.method == "PACE-SHAP-cond"]
    for d in ds:
        for setting in settings:
            g = one[(one.dataset == d) & (one.setting == setting)]
            if len(g):
                L.append(f"| {d} | {setting} | {g.eff_rows_large.mean():.1f} | "
                         f"{g.eff_rows_small.mean():.1f} | {g.off_share.mean():.3f} |")
    L += ["", "## Faithfulness by setting and method (mean ± 95% CI over seeds)", "",
          "| Dataset | Setting | Method | Cond. deletion | Cond. insertion | Interv. deletion | "
          "Interv. insertion | ms per explanation |", "|---|---|---|---|---|---|---|---|"]
    for (d, s, m), g in df.groupby(["dataset", "setting", "method"], sort=False):
        L.append(f"| {d} | {s} | {m} | {f(g.cond_margin_auc)} | {f(g.cond_insertion_margin_auc)} | "
                 f"{f(g.margin_auc)} | {f(g.insertion_margin_auc)} | {g.ms_per_explanation.mean():.1f} |")
    L.append("")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run D: the conditional test at more settings")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue a stopped run")
    ap.add_argument("--fresh", action="store_true", help="delete earlier results and start over")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if not a.summary_only:
        tasks = [(d, s) for d in a.datasets or DATASETS for s in a.seeds or SEEDS]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, a.jobs, COST)
    print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
