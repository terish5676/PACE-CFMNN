"""Test 2 of results/review_v1/PREREGISTRATION.md: the RQ1 review run (pre-registered).

Three questions from the pre-submission review about the v3 results, on new
seeds (50-59) with v3's models, splits, 50 points per dataset and tests:

* Stability (review M2). v3 re-drew KernelSHAP's background rows on every
  re-run but kept PACE-SHAP's. Here each method is re-run twice with a new
  sampling seed: once on its own rows ("fixed rows"), and once on 200 newly
  drawn training rows ("new rows").
* Ties (review M3). v3 divides each point's margin curve by the point's own
  margin, which is 0 for a tied point. Here the raw margin AUCs are added,
  and every AUC is also given on tied and untied points separately.
* LIME (review M5). v3 gave LIME the whole training set, not the explainer
  rows its pre-registration named. Here LIME uses the explainer rows, with
  v3's setting (no discretisation) and with LIME's default (quartile bins).

Results are saved after every dataset-seed pair.

Usage:
    python -B scripts/run_review_rq1.py --jobs 6             # the pre-registered run
    python -B scripts/run_review_rq1.py --jobs 6 --resume    # continue after a stop
    python -B scripts/run_review_rq1.py --summary-only       # rebuild summary.md
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
from pace_cfmnn.explain import PaceShap, pace_flip  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_error_detection import paired  # noqa: E402

SEEDS = list(range(50, 60))
OUT = ROOT / "results" / "review_v1" / "faithfulness"
METHODS = ["PACE-SHAP", "KernelSHAP", "LIME", "LIME-default", "PACE-flip", "Random"]
RERUN = 1000        # the second run's sampling seed is seed + RERUN, as in v3
MIN_UNTIED = 10     # a dataset enters the untied-point tests if every seed has this many
NEAR_TIE = 0.01
COST = {"S1": 5, "S2": 5, "S3": 5, "letter": 4, "landsat": 3, "magic": 2}


def explainer_rows(Xtr, seed, condition):
    """Explainer rows: v3's own ("fixed"), or 200 rows not used before ("new").

    New rows come from training rows outside both of v3's sets, so the tests
    never copy values from rows an explainer saw. Where fewer such rows exist
    than needed (Wine, Breast Cancer), they are a bootstrap resample of v3's
    explainer rows instead.
    """
    _, bex = rf.backgrounds_v3(Xtr, seed)
    if condition == "fixed":
        return bex
    k = len(bex)
    perm = np.random.default_rng(seed + 500).permutation(len(Xtr))     # as backgrounds_v3
    rest = perm[2 * k:]
    rng = np.random.default_rng(seed + 2000)
    if len(rest) >= k:
        return Xtr[rest[rng.choice(len(rest), k, replace=False)]]
    return bex[rng.integers(k, size=k)]


def attributions(model, Xte, rows, r, kernel_rows):
    """Every method's attributions on these explainer rows, with sampling seed r."""
    fns = {
        "PACE-SHAP": lambda: PaceShap(model, rows[:rf.SHAP_V3_R], n_permutations=rf.SHAP_V3_P,
                                      seed=r).explain(Xte),
        "KernelSHAP": lambda: xb.kernel_shap(model, Xte, kernel_rows, seed=r, target="margin"),
        "LIME": lambda: xb.lime_tabular(model, Xte, rows, seed=r, target="margin"),
        "LIME-default": lambda: xb.lime_tabular(model, Xte, rows, seed=r, target="margin",
                                                discretize=True),
        "PACE-flip": lambda: pace_flip(model, Xte, rows),
        "Random": lambda: np.random.default_rng(r).uniform(size=Xte.shape),
    }
    out, ms = {}, {}
    for m, fn in fns.items():
        t0 = time.perf_counter()
        out[m] = fn()
        ms[m] = 1000 * (time.perf_counter() - t0) / len(Xte)
    return out, ms


def top3_jaccard(a, b):
    """v3's stability: overlap of the top 3 features by |attribution| (Jaccard, mean over points)."""
    ta = np.argsort(-np.abs(a), axis=1, kind="stable")[:, :3]
    tb = np.argsort(-np.abs(b), axis=1, kind="stable")[:, :3]
    return float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(ta, tb)]))


def run_one(name, seed, n):
    t0 = time.perf_counter()
    _, Xtr, Xte, model, _ = rf.setup(name, seed, n)
    bg, bex = rf.backgrounds_v3(Xtr, seed)
    new = explainer_rows(Xtr, seed, "new")
    ks = rf._same_rows(bex, rf.SHAP_BG_ROWS, seed)           # v3's KernelSHAP rows for this seed
    first, ms = attributions(model, Xte, bex, seed, ks)
    fixed, _ = attributions(model, Xte, bex, seed + RERUN, ks)
    fresh, _ = attributions(model, Xte, new, seed + RERUN,
                            rf._same_rows(new, rf.SHAP_BG_ROWS, seed + RERUN))
    det = model.predict_details(Xte)
    mx = rf._margin(det, det.cls)
    tied, near = mx <= 1e-12, mx < NEAR_TIE
    part = lambda v, mask: float(v[mask].mean()) if mask.any() else np.nan   # noqa: E731
    rows = []
    for m in METHODS:
        attr = first[m]
        dk, dm, dr = rf.deletion_auc(model, Xte, attr, bg, seed=seed, signed=True, with_raw=True)
        ik, im, ir = rf.insertion_auc(model, Xte, attr, bg, seed=seed, with_raw=True)
        rows.append({"dataset": name, "seed": seed, "method": m, "n": len(Xte),
                     "n_tied": int(tied.sum()), "n_near_tie": int(near.sum()),
                     "deletion_auc": dk.mean(), "insertion_auc": ik.mean(),
                     "margin_auc": dm.mean(), "insertion_margin_auc": im.mean(),
                     "margin_auc_raw": dr.mean(), "insertion_margin_auc_raw": ir.mean(),
                     "margin_auc_untied": part(dm, ~tied),
                     "insertion_margin_auc_untied": part(im, ~tied),
                     "margin_auc_raw_untied": part(dr, ~tied),
                     "insertion_margin_auc_raw_untied": part(ir, ~tied),
                     "margin_auc_raw_tied": part(dr, tied),
                     "insertion_margin_auc_raw_tied": part(ir, tied),
                     "stability_fixed": top3_jaccard(attr, fixed[m]),
                     "stability_new": top3_jaccard(attr, fresh[m]),
                     "ms_per_explanation": ms[m]})
    s = {r["method"]: r for r in rows}
    print(f"{name} seed {seed}: tied {int(tied.sum())} of {len(Xte)}; stability fixed/new "
          f"PACE-SHAP {s['PACE-SHAP']['stability_fixed']:.2f}/{s['PACE-SHAP']['stability_new']:.2f}, "
          f"KernelSHAP {s['KernelSHAP']['stability_fixed']:.2f}/{s['KernelSHAP']['stability_new']:.2f} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return rows


def block(L, df, a, b, metric, lower, title):
    """A pre-registered comparison: paired over seeds, Wilcoxon, Holm across datasets."""
    t = paired(df, a, b, metric=metric)
    good = t["diff"] < 0 if lower else t["diff"] > 0
    sig = t["p_holm"] < 0.05
    L += ["", f"## {title}", "",
          f"{a} minus {b} in `{metric}`; better for {a}: {'negative' if lower else 'positive'}. "
          f"Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across the {len(t)} datasets. "
          f"**{a} is better on {int((sig & good).sum())}, level on {int((~sig).sum())} and worse on "
          f"{int((sig & ~good).sum())}.**", "",
          f"| Dataset | Difference | Seeds where {a} is higher | p | p (Holm) |", "|---|---|---|---|---|"]
    for _, r in t.iterrows():
        L.append(f"| {r.dataset} | {r['diff']:+.4f} ± {r.ci:.4f} | {r.wins} of {r.n} | {r.p:.3g} | "
                 f"{r.p_holm:.3g} |")


def summarise(df, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    meta = df[df.method == "PACE-SHAP"]
    untied_ok = [d for d, g in meta.groupby("dataset", sort=False)
                 if (g.n - g.n_tied).min() >= MIN_UNTIED]
    L = ["# Review Test 2 (pre-registered): stability, ties and LIME", "",
         "Rules: [`../PREREGISTRATION.md`](../PREREGISTRATION.md) (Test 2). "
         f"Seeds {seeds}; v3's models, splits and explained points (50 per dataset and seed, 36 on "
         "Wine). Deletion: lower is more faithful. Insertion: higher is more faithful.", "",
         "## Explained points that are tied", "",
         "| Dataset | Points | Tied (margin 0) | Margin below 0.01 |", "|---|---|---|---|"]
    for ds, g in meta.groupby("dataset", sort=False):
        L.append(f"| {ds} | {g.n.sum()} | {g.n_tied.sum() / g.n.sum():.0%} | "
                 f"{g.n_near_tie.sum() / g.n.sum():.0%} |")
    sub = df[df.dataset.isin(untied_ok)]
    for a, b, metric, lower, title, d in [
            ("PACE-SHAP", "KernelSHAP", "stability_fixed", False,
             "S1 (primary): stability when each method keeps its rows", df),
            ("PACE-SHAP", "KernelSHAP", "stability_new", False,
             "S2 (primary): stability when every method gets new rows", df),
            ("PACE-SHAP", "KernelSHAP", "margin_auc_raw", True, "F1a: raw deletion margin AUC", df),
            ("PACE-SHAP", "KernelSHAP", "insertion_margin_auc_raw", False,
             "F1b: raw insertion margin AUC", df),
            ("PACE-SHAP", "KernelSHAP", "margin_auc_untied", True,
             f"F2a: deletion margin AUC on untied points (datasets with at least {MIN_UNTIED} in every "
             "seed)", sub),
            ("PACE-SHAP", "KernelSHAP", "insertion_margin_auc_untied", False,
             f"F2b: insertion margin AUC on untied points (datasets with at least {MIN_UNTIED} in every "
             "seed)", sub),
            ("PACE-SHAP", "LIME", "margin_auc", True, "F3a: PACE-SHAP vs LIME on the explainer rows, "
             "deletion margin AUC", df),
            ("PACE-SHAP", "LIME", "insertion_margin_auc", False, "F3b: PACE-SHAP vs LIME on the "
             "explainer rows, insertion margin AUC", df),
            ("PACE-SHAP", "KernelSHAP", "margin_auc", True,
             "Secondary: v3's primary test on the new seeds, deletion margin AUC", df),
            ("PACE-SHAP", "KernelSHAP", "insertion_margin_auc", False,
             "Secondary: v3's primary test on the new seeds, insertion margin AUC", df),
            ("LIME-default", "LIME", "margin_auc", True,
             "Descriptive: LIME's default discretisation vs v3's setting, deletion margin AUC", df)]:
        if len(d):
            block(L, d, a, b, metric, lower, title)
    L += ["", "## Means over seeds (descriptive)"]
    for metric, title in [("stability_fixed", "Stability, rows kept"),
                          ("stability_new", "Stability, new rows"),
                          ("margin_auc", "Deletion margin AUC (normalised, as v3)"),
                          ("insertion_margin_auc", "Insertion margin AUC (normalised, as v3)"),
                          ("margin_auc_raw", "Deletion margin AUC (raw)"),
                          ("insertion_margin_auc_raw", "Insertion margin AUC (raw)"),
                          ("ms_per_explanation", "Milliseconds per explanation (runs share the CPU)")]:
        L += ["", f"### {title}", "", "| Dataset | " + " | ".join(METHODS) + " |",
              "|---|" + "---|" * len(METHODS)]
        for ds, g in df.groupby("dataset", sort=False):
            cells = [g[g.method == m][metric] for m in METHODS]
            L.append(f"| {ds} | " + " | ".join(f"{v.mean():.3f}" if v.notna().any() else "–"
                                              for v in cells) + " |")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Review Test 2: stability, ties and LIME (pre-registered)")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--n", type=int, default=50, help="explained points per dataset and seed")
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
        tasks = [(d, s, a.n) for d in a.datasets or rf.DATASETS for s in a.seeds or SEEDS]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, a.jobs, COST)
    print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
