"""Run C: RQ1 with 200 points per seed, and tie-aware outputs.

Rules: results/rq1_more_points/PREREGISTRATION.md. Plan: research_audit/LIMITATIONS_PLAN.md, run C.

Protocol v3 of `scripts/run_faithfulness.py` (separate explainer background, signed ranking, margin
targets, 10 repetitions), with all its methods and settings, on fresh seeds (110-119) and 200 test
points per seed instead of 50 (Wine keeps all 36 of its test rows). One method is added:

* KernelSHAP-same: KernelSHAP (margin target) on PACE-SHAP's own 200 explainer rows
  (`run_faithfulness._same_rows`), the like-for-like arm. v3's KernelSHAP uses 20 of them.

Tie-aware outputs, as in `scripts/run_review_rq1.py`: the number of tied points (margin <= 1e-12)
and near-tied points (margin < 0.01); the AUCs on untied points only; the raw, undivided margin
AUCs; and the class-kept AUCs.

    python -B scripts/run_more_points.py --jobs 12
    python -B scripts/run_more_points.py --jobs 12 --resume
    python -B scripts/run_more_points.py --summary-only
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
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_error_detection import ci95, paired  # noqa: E402
from run_review_rq1 import MIN_UNTIED, NEAR_TIE, top3_jaccard  # noqa: E402

OUT = ROOT / "results" / "rq1_more_points"
DATASETS = rf.DATASETS
SEEDS = list(range(110, 120))
N_POINTS = 200
RERUN = 1000                      # the stability re-run's sampling seed is seed + 1000, as in v3
# v3's methods in v3's order, then the added arm last (so v3's methods keep their random streams)
METHODS = ["PACE-SHAP", "PACE-SHAP-20", "PACE-flip", "PACE-multi", "PACE-contrastive",
           "PACE-decomposition", "KernelSHAP", "KernelSHAP-strong", "LIME", "Random", "KernelSHAP-same"]
# (focus, other, metric, lower is better): the pre-registered primaries
PRIMARIES = [("PACE-SHAP", "KernelSHAP", "margin_auc", True),
             ("PACE-SHAP", "KernelSHAP", "insertion_margin_auc", False),
             ("PACE-SHAP", "KernelSHAP-same", "margin_auc", True),
             ("PACE-SHAP", "KernelSHAP-same", "insertion_margin_auc", False)]
# Slowest first: this only orders the tasks.
COST = {"letter": 6, "magic": 5, "landsat": 4, "shuttle": 3, "page_blocks": 3, "thyroid": 2}


def methods_c(model, Xte, Xtr, bex, seed):
    """v3's methods, plus KernelSHAP on PACE-SHAP's own explainer rows."""
    out = dict(rf.methods_v3(model, Xte, Xtr, bex))
    same = rf._same_rows(bex, rf.SHAP_V3_R, seed)
    out["KernelSHAP-same"] = lambda r: xb.kernel_shap(model, Xte, same, seed=r, target="margin")
    return {m: out[m] for m in METHODS}


def run_one(name: str, seed: int, n: int = N_POINTS):
    t_start = time.perf_counter()
    ds, Xtr, Xte, model, _ = rf.setup(name, seed, n)
    bg, bex = rf.backgrounds_v3(Xtr, seed)
    relevant = set(ds.info.get("relevant", [])) if name in ("S1", "S2", "S3") else None
    det = model.predict_details(Xte)
    mx = rf._margin(det, det.cls)
    tied, near = mx <= 1e-12, mx < NEAR_TIE
    part = lambda v, mask: float(v[mask].mean()) if mask.any() else np.nan   # noqa: E731
    gt_rng = np.random.default_rng(seed + 7)
    rows = []
    for m, fn in methods_c(model, Xte, Xtr, bex, seed).items():
        t0 = time.perf_counter()
        attr = fn(seed)
        dt = (time.perf_counter() - t0) / len(Xte)
        stab = np.nan if m in rf.NO_STABILITY_RERUN else top3_jaccard(attr, fn(seed + RERUN))
        dk, dm, dr = rf.deletion_auc(model, Xte, attr, bg, seed=seed, signed=True, with_raw=True)
        _, dm_abs = rf.deletion_auc(model, Xte, attr, bg, seed=seed, signed=False)
        ik, im, ir = rf.insertion_auc(model, Xte, attr, bg, seed=seed, with_raw=True)
        t3 = rf.top3_random_ties(attr, gt_rng)
        gt = float(np.mean([len(set(t) & relevant) / 3 for t in t3])) if relevant else np.nan
        rows.append({"dataset": name, "seed": seed, "method": m, "protocol": "v3-200", "n": len(Xte),
                     "n_tied": int(tied.sum()), "n_near_tie": int(near.sum()),
                     "deletion_auc": dk.mean(), "margin_auc": dm.mean(), "margin_auc_abs": dm_abs.mean(),
                     "insertion_auc": ik.mean(), "insertion_margin_auc": im.mean(),
                     "margin_auc_raw": dr.mean(), "insertion_margin_auc_raw": ir.mean(),
                     "margin_auc_untied": part(dm, ~tied), "insertion_margin_auc_untied": part(im, ~tied),
                     "margin_auc_raw_untied": part(dr, ~tied),
                     "insertion_margin_auc_raw_untied": part(ir, ~tied),
                     "gt_top3_precision": gt, "share_negative": float((attr < 0).mean()),
                     "stability_top3": stab, "ms_per_explanation": 1000 * dt})
    print(f"{name} seed {seed}: {len(Xte)} points, tied {int(tied.sum())}, "
          f"{time.perf_counter() - t_start:.0f} s", flush=True)
    return rows


# ------------------------------------------------------------------ summary
def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    ds = [d for d in DATASETS if d in set(df.dataset)]
    methods = [m for m in METHODS if m in set(df.method)]
    f = lambda v: f"{v.mean():.3f} ± {ci95(v):.3f}"   # noqa: E731
    L = ["# Run C: RQ1 with 200 points per seed, tie-aware", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Protocol v3, seeds "
         f"{seeds[0]}–{seeds[-1]} ({len(seeds)}), {int(df.n.max())} test points per dataset and seed "
         "(fewer if the test split is smaller). Deletion margin AUC: lower is more faithful. "
         "Insertion margin AUC: higher is more faithful. Each comparison: paired over seeds, "
         "two-sided Wilcoxon signed-rank test, Holm across datasets. Written by "
         "`scripts/run_more_points.py`.", "",
         "## Primary comparisons", ""]

    def block(a, b, metric, lower, heading=None, data=None):
        res = paired(df if data is None else data, a, b, metric=metric)
        res = res[res["n"] > 0]
        good = res["diff"] < 0 if lower else res["diff"] > 0
        sig = res["p_holm"] < 0.05
        L.extend([f"**{heading or f'{a} minus {b}, {metric}'}** ({'negative' if lower else 'positive'} "
                  f"favours {a}): {a} is better on {int((sig & good).sum())}, level on "
                  f"{int((~sig).sum())}, worse on {int((sig & ~good).sum())} of {len(res)} datasets.", "",
                  f"| Dataset | Difference | Seeds where {a} is higher | p | p (Holm) | Verdict |",
                  "|---|---|---|---|---|---|"])
        for (_, r), g, s in zip(res.iterrows(), good, sig):
            v = "level" if not s else "better" if g else "worse"
            L.append(f"| {r.dataset} | {r['diff']:+.4f} ± {r.ci:.4f} | {r.wins} of {r.n} | "
                     f"{r.p:.3g} | {r.p_holm:.3g} | {v} |")
        L.append("")

    for a, b, metric, lower in PRIMARIES:
        block(a, b, metric, lower)
    meta = df.drop_duplicates(["dataset", "seed"])
    ties = meta.groupby("dataset", sort=False).agg(points=("n", "mean"), tied=("n_tied", "mean"),
                                                   near=("n_near_tie", "mean")).reindex(ds)
    ties["tied_share_%"] = 100 * ties.tied / ties.points
    ties["near_share_%"] = 100 * ties.near / ties.points
    L += ["## Tied and near-tied points (margin <= 1e-12, margin < 0.01), mean over seeds", "", "```",
          ties.round(1).to_string(), "```", ""]
    enough = [d for d in ds if (meta[meta.dataset == d].n - meta[meta.dataset == d].n_tied).min() >= MIN_UNTIED]
    L += [f"## The same comparisons on untied points only (datasets with at least {MIN_UNTIED} untied "
          f"points in every seed: {', '.join(enough) or 'none'})", ""]
    sub = df[df.dataset.isin(enough)]
    for a, b, metric, lower in PRIMARIES:
        if len(sub):
            block(a, b, metric + "_untied", lower, f"{a} minus {b}, {metric}, untied points", data=sub)
    L += ["## The same comparisons on the raw (undivided) margin AUCs", ""]
    for a, b, metric, lower in PRIMARIES:
        block(a, b, metric + "_raw", lower, f"{a} minus {b}, {metric} (raw)")
    L += ["## The class-kept AUCs (deletion: lower is more faithful; insertion: higher)", ""]
    for a, b in (("PACE-SHAP", "KernelSHAP"), ("PACE-SHAP", "KernelSHAP-same")):
        block(a, b, "deletion_auc", True, f"{a} minus {b}, class-kept deletion AUC")
        block(a, b, "insertion_auc", False, f"{a} minus {b}, class-kept insertion AUC")
    L += ["## All methods: deletion and insertion margin AUC, stability, milliseconds", "",
          "| Dataset | Method | Deletion margin AUC | Insertion margin AUC | Class-kept deletion AUC | "
          "Stability | ms per explanation |", "|---|---|---|---|---|---|---|"]
    for (d, m), g in df.groupby(["dataset", "method"], sort=False):
        stab = "–" if g.stability_top3.isna().all() else f"{g.stability_top3.mean():.2f}"
        L.append(f"| {d} | {m} | {f(g.margin_auc)} | {f(g.insertion_margin_auc)} | {f(g.deletion_auc)} | "
                 f"{stab} | {g.ms_per_explanation.mean():.2f} |")
    L.append("")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run C: RQ1 with 200 points, tie-aware")
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
